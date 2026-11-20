"""Training entry point organized after the original src/OASIS/trainer.py."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from data_utils import CyclingLoader, build_dataloader, build_paired_anchor_dataloader, group_microbatches
from finetune import (
    OASISConfig,
    OASISState,
    OASISTrainer,
    collect_response_activations,
    fit_initial_direction,
)


CONFIG_DEFAULTS = {
    "gradient_accumulation_steps": 1,
    "gradient_checkpointing": True,
    "direction_lr": 1e-4,
    "direction_steps": 1,
    "initial_direction_lr": 1e-2,
    "initial_direction_steps": 1000,
    "logging_steps": 1,
}


def load_config(path: str | Path, setting: str) -> SimpleNamespace:
    with open(path, encoding="utf-8") as handle:
        configs = json.load(handle)
    if setting not in configs:
        raise KeyError(f"unknown setting {setting!r}; choose from {list(configs)}")
    values = CONFIG_DEFAULTS | configs[setting]
    values.setdefault("per_device_train_batch_size", values["batch_size"])
    return SimpleNamespace(**values)


def load_model_and_tokenizer(config):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("install the training dependencies with pip install -r requirements.txt") from error

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config.model, torch_dtype=dtype)
    if getattr(config, "gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as error:
        raise RuntimeError("PEFT is required for the paper LoRA setup: pip install peft") from error
    lora = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.target_modules,
    )
    model = get_peft_model(model, lora)
    # LoRA freezes the embedding parameters. Checkpointed transformer blocks
    # still need a grad-carrying input so adapter gradients are retained.
    if getattr(config, "gradient_checkpointing", True) and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return model, tokenizer


def _load_initial_direction(path: str | Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu")
    if isinstance(value, torch.Tensor):
        return value.reshape(-1)
    if isinstance(value, dict) and "direction" in value:
        return value["direction"].reshape(-1)
    raise ValueError("initial_direction_path must contain a Tensor or {'direction': Tensor}")


def train(config) -> None:
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer(config)
    model.to(device)

    train_loader = build_dataloader(
        config.training_file,
        tokenizer,
        batch_size=config.per_device_train_batch_size,
        max_length=config.max_seq_length,
        shuffle=True,
        drop_last=True,
    )
    effective_batch_size = config.per_device_train_batch_size * config.gradient_accumulation_steps
    if effective_batch_size != config.batch_size:
        raise ValueError(
            f"microbatch * accumulation must equal paper batch size: {effective_batch_size} != {config.batch_size}"
        )
    if config.batch_size % 2:
        raise ValueError("balanced anchors require an even effective batch size")

    anchor_batch_size = config.batch_size // 2
    positive_loader = build_dataloader(
        config.contrastive_positive_file,
        tokenizer,
        batch_size=anchor_batch_size,
        max_length=config.max_seq_length,
        shuffle=False,
        drop_last=False,
    )
    negative_loader = build_dataloader(
        config.contrastive_negative_file,
        tokenizer,
        batch_size=anchor_batch_size,
        max_length=config.max_seq_length,
        shuffle=False,
        drop_last=False,
    )

    if config.initial_direction_path:
        initial_direction = _load_initial_direction(config.initial_direction_path)
    else:
        positive_activations = collect_response_activations(
            model, positive_loader, layer=config.intervention_layer, device=device
        )
        negative_activations = collect_response_activations(
            model, negative_loader, layer=config.intervention_layer, device=device
        )
        initial_direction = fit_initial_direction(
            positive_activations.to(device),
            negative_activations.to(device),
            margin=config.margin,
            l2=config.l2,
            lr=config.initial_direction_lr,
            steps=config.initial_direction_steps,
        ).cpu()
        output = Path(config.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        torch.save(initial_direction, output / "initial_direction.pt")

    oasis_config = OASISConfig(
        tau_positive=config.tau_positive,
        tau_negative=config.tau_negative,
        margin=config.margin,
        l2=config.l2,
        direction_lr=config.direction_lr,
        direction_steps=config.direction_steps,
        ema_rate=config.ema_rate,
        schedule_start_fraction=config.schedule_start_fraction,
        certainty_min=config.certainty_min,
        max_interval=config.max_interval,
    )
    oasis = OASISState(initial_direction, oasis_config)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    steps_per_epoch = len(train_loader) // config.gradient_accumulation_steps
    total_steps = steps_per_epoch * config.epochs
    if total_steps < 1:
        raise ValueError("training data is smaller than one effective batch")
    try:
        from transformers import get_scheduler

        lr_scheduler = get_scheduler(
            config.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=math.ceil(total_steps * config.warmup_ratio),
            num_training_steps=total_steps,
        )
    except ImportError:
        lr_scheduler = None

    engine = OASISTrainer(
        model=model,
        optimizer=optimizer,
        oasis=oasis,
        intervention_layer=config.intervention_layer,
        device=device,
        lr_scheduler=lr_scheduler,
    )
    paired_anchor_loader = build_paired_anchor_dataloader(
        config.contrastive_positive_file,
        config.contrastive_negative_file,
        tokenizer,
        batch_size=anchor_batch_size,
        max_length=config.max_seq_length,
    )
    anchor_cycle = CyclingLoader(paired_anchor_loader)
    anchor_batches = lambda: next(anchor_cycle)

    global_step = 0
    for epoch in range(config.epochs):
        for microbatches in group_microbatches(train_loader, config.gradient_accumulation_steps):
            global_step += 1
            metrics = engine.train_effective_batch(
                microbatches,
                anchor_batches=anchor_batches,
                step=global_step,
                total_steps=total_steps,
            )
            if global_step % config.logging_steps == 0:
                print(
                    f"epoch={epoch + 1} step={global_step}/{total_steps} "
                    f"loss={metrics.loss:.6f} certainty={metrics.certainty:.4f} "
                    f"direction_updated={metrics.direction_updated} interval={metrics.update_interval}"
                )
            if config.save_steps and global_step % config.save_steps == 0:
                checkpoint = Path(config.output_dir) / f"checkpoint-{global_step}"
                model.save_pretrained(checkpoint)
                tokenizer.save_pretrained(checkpoint)
                engine.save_checkpoint(checkpoint, step=global_step)

    output = Path(config.output_dir)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    engine.save_checkpoint(output, step=global_step)
    # Steering hooks are scoped to each forward in OASISTrainer and are absent here.
    print(f"Training complete. Intervention-free model saved to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="OASIS LoRA fine-tuning")
    parser.add_argument("--config", default="train_config.json")
    parser.add_argument("--setting", default="Setting_1")
    args = parser.parse_args()
    train(load_config(args.config, args.setting))


if __name__ == "__main__":
    main()
