"""OASIS algorithm and training engine.

The implementation follows method.tex directly: response-only pooling,
margin-softplus direction learning, balanced pseudo-labels, forward-only
anchors, EMA, adaptive update scheduling, and example-wise steering.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class OASISConfig:
    tau_positive: float = 0.8
    tau_negative: float = 0.2
    margin: float = 1.0
    l2: float = 1e-3
    direction_lr: float = 1e-4
    direction_steps: int = 1
    ema_rate: float = 0.25
    schedule_start_fraction: float = 0.25
    certainty_min: float = 0.85
    max_interval: int = 50

    def __post_init__(self) -> None:
        if not 0 <= self.tau_negative < self.tau_positive <= 1:
            raise ValueError("require 0 <= tau_negative < tau_positive <= 1")
        if self.margin < 0 or self.l2 < 0 or self.direction_lr <= 0:
            raise ValueError("invalid direction optimization hyperparameters")
        if self.direction_steps < 1 or not 0 < self.ema_rate <= 1:
            raise ValueError("direction_steps >= 1 and 0 < ema_rate <= 1 are required")
        if not 0 <= self.schedule_start_fraction <= 1:
            raise ValueError("schedule_start_fraction must be in [0, 1]")
        if not 0.5 <= self.certainty_min <= 1 or self.max_interval < 1:
            raise ValueError("invalid scheduler hyperparameters")


def response_mask_from_batch(batch: Mapping[str, Any]) -> Tensor:
    if "response_mask" in batch:
        mask = batch["response_mask"].bool()
    elif "labels" in batch:
        mask = batch["labels"].ne(-100)
    else:
        raise ValueError("each batch needs response_mask or response-only labels")
    if "attention_mask" in batch:
        mask = mask & batch["attention_mask"].bool()
    if not torch.all(mask.any(dim=1)):
        raise ValueError("every sample must contain at least one response token")
    return mask


def model_inputs(batch: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in batch.items() if key != "response_mask"}


def response_average(activations: Tensor, response_mask: Tensor) -> Tensor:
    if activations.ndim != 3 or response_mask.shape != activations.shape[:2]:
        raise ValueError("expected activations [B,T,D] and response_mask [B,T]")
    mask = response_mask.to(device=activations.device, dtype=torch.bool)
    counts = mask.sum(dim=1)
    if torch.any(counts == 0):
        raise ValueError("every sample must contain at least one response token")
    return (activations * mask.unsqueeze(-1)).sum(dim=1) / counts.unsqueeze(-1)


def margin_softplus_loss(
    direction: Tensor,
    activations: Tensor,
    labels: Tensor,
    *,
    margin: float,
    l2: float,
) -> Tensor:
    labels = labels.to(device=activations.device, dtype=activations.dtype).reshape(-1)
    if activations.ndim != 2 or labels.shape[0] != activations.shape[0]:
        raise ValueError("expected activations [N,D] and labels [N]")
    if not torch.all((labels == 1) | (labels == -1)):
        raise ValueError("labels must be +1 or -1")
    scores = labels * (activations @ direction.reshape(-1))
    return F.softplus(float(margin) - scores).mean() + 0.5 * float(l2) * direction.square().sum()


@torch.no_grad()
def direction_sgd(
    initial: Tensor,
    activations: Tensor,
    labels: Tensor,
    *,
    margin: float,
    l2: float,
    lr: float,
    steps: int,
) -> Tensor:
    x = activations.detach().float()
    y = labels.detach().to(device=x.device, dtype=torch.float32).reshape(-1)
    v = initial.detach().to(device=x.device, dtype=torch.float32).reshape(-1).clone()
    for _ in range(int(steps)):
        weights = torch.sigmoid(float(margin) - y * (x @ v))
        gradient = x.t().mv(-y * weights) / x.shape[0] + float(l2) * v
        v.add_(gradient, alpha=-float(lr))
    return v.to(dtype=initial.dtype)


def fit_initial_direction(
    positive: Tensor,
    negative: Tensor,
    *,
    margin: float = 1.0,
    l2: float = 1e-3,
    lr: float = 1e-2,
    steps: int = 1000,
) -> Tensor:
    if positive.ndim != 2 or negative.ndim != 2 or positive.shape[1] != negative.shape[1]:
        raise ValueError("positive and negative activations must be [N,D]")
    x = torch.cat((positive, negative), dim=0)
    y = torch.cat((torch.ones(len(positive)), -torch.ones(len(negative)))).to(x.device)
    return direction_sgd(
        torch.zeros(x.shape[1], device=x.device, dtype=x.dtype),
        x,
        y,
        margin=margin,
        l2=l2,
        lr=lr,
        steps=steps,
    )


class DirectionUpdateScheduler:
    def __init__(self, config: OASISConfig):
        self.config = config
        self.stable_streak = 0
        self.next_update_step = 1
        self.last_step = 0

    def decide(self, probabilities: Tensor, *, step: int, total_steps: int) -> tuple[bool, float, int]:
        if step != self.last_step + 1:
            raise ValueError(f"expected OASIS step {self.last_step + 1}, got {step}")
        if total_steps < step:
            raise ValueError("step cannot exceed total_steps")
        self.last_step = step
        certainty = torch.maximum(probabilities, 1 - probabilities).mean().item()
        self.stable_streak = self.stable_streak + 1 if certainty >= self.config.certainty_min else 0
        early = step / total_steps <= self.config.schedule_start_fraction
        interval = 1 if early else min(
            self.config.max_interval,
            max(1, 10 * (self.stable_streak // 10)),
        )
        update = early or step == self.next_update_step
        if update:
            self.next_update_step = step + interval
        return update, float(certainty), interval

    def state_dict(self) -> dict[str, int]:
        return {
            "stable_streak": self.stable_streak,
            "next_update_step": self.next_update_step,
            "last_step": self.last_step,
        }

    def load_state_dict(self, state: Mapping[str, int]) -> None:
        self.stable_streak = int(state["stable_streak"])
        self.next_update_step = int(state["next_update_step"])
        self.last_step = int(state["last_step"])


@dataclass
class StepMetrics:
    loss: float
    direction_updated: bool
    certainty: float
    update_interval: int
    pseudo_positive: int
    pseudo_negative: int


class OASISState:
    def __init__(self, initial_direction: Tensor, config: OASISConfig):
        if initial_direction.ndim != 1:
            raise ValueError("initial_direction must be [D]")
        self.direction = initial_direction.detach().clone()
        self.config = config
        self.scheduler = DirectionUpdateScheduler(config)

    def probabilities(self, pooled: Tensor) -> Tensor:
        return torch.sigmoid(pooled.float() @ self.direction.to(pooled.device).float())

    @torch.no_grad()
    def update(
        self,
        pooled_batch: Tensor,
        *,
        step: int,
        total_steps: int,
        anchor_provider: Callable[[], tuple[Tensor, Tensor]],
        generator: torch.Generator | None = None,
    ) -> tuple[bool, float, int, int, int]:
        probabilities = self.probabilities(pooled_batch)
        update, certainty, interval = self.scheduler.decide(
            probabilities, step=step, total_steps=total_steps
        )
        pos_idx = torch.nonzero(probabilities >= self.config.tau_positive).flatten()
        neg_idx = torch.nonzero(probabilities <= self.config.tau_negative).flatten()
        n_pos, n_neg = int(pos_idx.numel()), int(neg_idx.numel())
        if not update:
            return False, certainty, interval, n_pos, n_neg

        n = min(n_pos, n_neg)
        if n:
            pos_idx = pos_idx[torch.randperm(n_pos, device=pos_idx.device, generator=generator)[:n]]
            neg_idx = neg_idx[torch.randperm(n_neg, device=neg_idx.device, generator=generator)[:n]]
            pseudo_x = torch.cat((pooled_batch[pos_idx], pooled_batch[neg_idx]))
            pseudo_y = torch.cat((pooled_batch.new_ones(n), -pooled_batch.new_ones(n)))
        else:
            pseudo_x = pooled_batch.new_empty((0, pooled_batch.shape[1]))
            pseudo_y = pooled_batch.new_empty((0,))

        anchor_positive, anchor_negative = anchor_provider()
        if anchor_positive.shape[0] != anchor_negative.shape[0]:
            raise ValueError("positive and negative anchor counts must be equal")
        if anchor_positive.shape[0] + anchor_negative.shape[0] != pooled_batch.shape[0]:
            raise ValueError("total anchor-set size must equal the effective per-step batch size")
        x = torch.cat((pseudo_x, anchor_positive, anchor_negative))
        y = torch.cat(
            (
                pseudo_y,
                pooled_batch.new_ones(anchor_positive.shape[0]),
                -pooled_batch.new_ones(anchor_negative.shape[0]),
            )
        )
        old = self.direction.to(x.device)
        candidate = direction_sgd(
            old,
            x,
            y,
            margin=self.config.margin,
            l2=self.config.l2,
            lr=self.config.direction_lr,
            steps=self.config.direction_steps,
        )
        self.direction = (
            (1 - self.config.ema_rate) * old + self.config.ema_rate * candidate
        ).detach()
        return True, certainty, interval, n_pos, n_neg

    def steer(self, activations: Tensor, response_mask: Tensor) -> tuple[Tensor, Tensor]:
        pooled = response_average(activations, response_mask)
        direction = self.direction.to(device=activations.device, dtype=activations.dtype)
        alpha = torch.sigmoid(pooled.float() @ direction.float()).to(activations.dtype)
        delta = alpha[:, None, None] * direction[None, None, :]
        mask = response_mask.to(device=activations.device, dtype=activations.dtype).unsqueeze(-1)
        return activations + mask * delta, alpha

    def state_dict(self) -> dict[str, Any]:
        return {
            "direction": self.direction.cpu(),
            "config": asdict(self.config),
            "scheduler": self.scheduler.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.direction = state["direction"].detach().clone()
        self.scheduler.load_state_dict(state["scheduler"])


def resolve_submodule(model: nn.Module, path: str) -> nn.Module:
    candidates = [path, f"model.{path}", f"base_model.{path}", f"base_model.model.{path}"]
    for candidate in dict.fromkeys(candidates):
        try:
            return model.get_submodule(candidate)
        except AttributeError:
            continue
    raise ValueError(f"cannot resolve intervention layer {path!r}; tried {candidates}")


def _output_tensor(output: Any) -> Tensor:
    return output[0] if isinstance(output, tuple) else output


def _replace_output(output: Any, tensor: Tensor) -> Any:
    return (tensor, *output[1:]) if isinstance(output, tuple) else tensor


@contextmanager
def capture_hook(model: nn.Module, layer: str, response_mask: Tensor):
    captured: list[Tensor] = []

    def hook(_module, _inputs, output):
        captured.append(response_average(_output_tensor(output), response_mask).detach())

    handle = resolve_submodule(model, layer).register_forward_hook(hook)
    try:
        yield captured
    finally:
        handle.remove()


@contextmanager
def steering_hook(model: nn.Module, layer: str, response_mask: Tensor, oasis: OASISState):
    def hook(_module, _inputs, output):
        steered, _ = oasis.steer(_output_tensor(output), response_mask)
        return _replace_output(output, steered)

    handle = resolve_submodule(model, layer).register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def move_batch(batch: Mapping[str, Any], device: torch.device | str) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


class OASISTrainer:
    """Two-pass trainer whose OASIS step is one effective optimizer batch.

    Probe all microbatches first, update v once, then replay the same microbatches
    with steering and gradients. This preserves method.tex when gradient
    accumulation is used.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        oasis: OASISState,
        intervention_layer: str,
        device: torch.device | str,
        lr_scheduler: Any | None = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.oasis = oasis
        self.intervention_layer = intervention_layer
        self.device = device
        self.lr_scheduler = lr_scheduler

    @torch.no_grad()
    def capture_batch(self, batch: Mapping[str, Any]) -> Tensor:
        batch = move_batch(batch, self.device)
        mask = response_mask_from_batch(batch)
        with capture_hook(self.model, self.intervention_layer, mask) as captured:
            self.model(**model_inputs(batch))
        if len(captured) != 1:
            raise RuntimeError(f"intervention layer executed {len(captured)} times")
        return captured[0]

    def train_effective_batch(
        self,
        microbatches: Sequence[Mapping[str, Any]],
        *,
        anchor_batches: Callable[[], tuple[Mapping[str, Any], Mapping[str, Any]]],
        step: int,
        total_steps: int,
    ) -> StepMetrics:
        if not microbatches:
            raise ValueError("microbatches must not be empty")
        self.model.train()
        # Save RNG state before each probe so dropout produces the same unsteered
        # activation again in the gradient-bearing replay.
        pooled_chunks: list[Tensor] = []
        rng_states: list[tuple[Tensor, list[Tensor] | None]] = []
        for batch in microbatches:
            cpu_state = torch.random.get_rng_state()
            cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            rng_states.append((cpu_state, cuda_state))
            pooled_chunks.append(self.capture_batch(batch))
        pooled_batch = torch.cat(pooled_chunks)

        def capture_anchors() -> tuple[Tensor, Tensor]:
            positive, negative = anchor_batches()
            return self.capture_batch(positive), self.capture_batch(negative)

        updated, certainty, interval, n_pos, n_neg = self.oasis.update(
            pooled_batch,
            step=step,
            total_steps=total_steps,
            anchor_provider=capture_anchors,
        )

        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for batch, (cpu_state, cuda_state) in zip(microbatches, rng_states):
            torch.random.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
            batch = move_batch(batch, self.device)
            mask = response_mask_from_batch(batch)
            # Backward occurs inside the context, so checkpoint recomputation is steered too.
            with steering_hook(self.model, self.intervention_layer, mask, self.oasis):
                output = self.model(**model_inputs(batch))
                loss = output.loss if hasattr(output, "loss") else output[0]
                (loss / len(microbatches)).backward()
                total_loss += float(loss.detach().item())
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return StepMetrics(
            loss=total_loss / len(microbatches),
            direction_updated=updated,
            certainty=certainty,
            update_interval=interval,
            pseudo_positive=n_pos,
            pseudo_negative=n_neg,
        )

    def save_checkpoint(self, directory: str | Path, *, step: int) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "step": step,
                "oasis": self.oasis.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": None if self.lr_scheduler is None else self.lr_scheduler.state_dict(),
            },
            directory / "oasis_training_state.pt",
        )

    def load_checkpoint(self, path: str | Path) -> int:
        state = torch.load(path, map_location="cpu")
        self.oasis.load_state_dict(state["oasis"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.lr_scheduler is not None and state["lr_scheduler"] is not None:
            self.lr_scheduler.load_state_dict(state["lr_scheduler"])
        return int(state["step"])


@torch.no_grad()
def collect_response_activations(
    model: nn.Module,
    batches: Iterable[Mapping[str, Any]],
    *,
    layer: str,
    device: torch.device | str,
) -> Tensor:
    was_training = model.training
    model.eval()
    chunks = []
    try:
        for batch in batches:
            batch = move_batch(batch, device)
            mask = response_mask_from_batch(batch)
            with capture_hook(model, layer, mask) as captured:
                model(**model_inputs(batch))
            chunks.append(captured[0].cpu())
    finally:
        model.train(was_training)
    if not chunks:
        raise ValueError("activation dataloader is empty")
    return torch.cat(chunks)
