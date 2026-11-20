"""Datasets and collators for OASIS fine-tuning and contrastive anchors."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def split_prompt_response(messages: Sequence[Mapping[str, Any]]) -> tuple[list[dict], dict]:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.get("role") == "assistant" and isinstance(message.get("content"), str):
            return [dict(item) for item in messages[:index]], dict(message)
    raise ValueError("each example needs a final assistant response")


class ChatDataset(Dataset):
    """Tokenize JSONL chat examples and mask prompt tokens in labels."""

    def __init__(self, path: str | Path, tokenizer, *, max_length: int = 2048):
        self.rows = load_jsonl(path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        messages = self.rows[index]["messages"]
        prompt_messages, _ = split_prompt_response(messages)
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]
        encoded = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask", [1] * len(input_ids))
        prompt_length = min(len(prompt_ids), len(input_ids))
        labels = [-100] * prompt_length + input_ids[prompt_length:]
        if not any(label != -100 for label in labels):
            raise ValueError(f"example {index} has no response tokens after truncation")
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class ResponseCollator:
    def __init__(self, tokenizer, *, pad_to_multiple_of: int | None = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, examples: Sequence[Mapping[str, list[int]]]) -> dict[str, Tensor]:
        features = [
            {"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]}
            for item in examples
        ]
        batch = self.tokenizer.pad(
            features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        max_length = batch["input_ids"].shape[1]
        labels = [item["labels"] + [-100] * (max_length - len(item["labels"])) for item in examples]
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch


class PairedChatDataset(Dataset):
    """Keep positive/negative contrastive examples aligned by row index."""

    def __init__(self, positive_path, negative_path, tokenizer, *, max_length: int):
        self.positive = ChatDataset(positive_path, tokenizer, max_length=max_length)
        self.negative = ChatDataset(negative_path, tokenizer, max_length=max_length)
        if len(self.positive) != len(self.negative):
            raise ValueError("paired positive/negative anchor files must have equal length")

    def __len__(self) -> int:
        return len(self.positive)

    def __getitem__(self, index: int):
        return self.positive[index], self.negative[index]


class PairedResponseCollator:
    def __init__(self, tokenizer):
        self.collator = ResponseCollator(tokenizer)

    def __call__(self, examples):
        positive, negative = zip(*examples)
        return self.collator(positive), self.collator(negative)


def build_dataloader(
    path: str | Path,
    tokenizer,
    *,
    batch_size: int,
    max_length: int,
    shuffle: bool,
    drop_last: bool = False,
) -> DataLoader:
    dataset = ChatDataset(path, tokenizer, max_length=max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=ResponseCollator(tokenizer),
    )


def build_paired_anchor_dataloader(
    positive_path: str | Path,
    negative_path: str | Path,
    tokenizer,
    *,
    batch_size: int,
    max_length: int,
    shuffle: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    dataset = PairedChatDataset(
        positive_path, negative_path, tokenizer, max_length=max_length
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=PairedResponseCollator(tokenizer),
    )


class CyclingLoader:
    """Cycle a non-empty dataloader indefinitely."""

    def __init__(self, loader: Iterable[Mapping[str, Tensor]]):
        self.loader = loader
        self.iterator: Iterator[Mapping[str, Tensor]] = iter(loader)

    def __next__(self) -> Mapping[str, Tensor]:
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            try:
                return next(self.iterator)
            except StopIteration as error:
                raise ValueError("anchor dataloader is empty") from error


def group_microbatches(
    loader: Iterable[Mapping[str, Tensor]], accumulation_steps: int
) -> Iterator[list[Mapping[str, Tensor]]]:
    """Yield only complete optimizer-step batches for reproducible batch size."""
    group: list[Mapping[str, Tensor]] = []
    for batch in loader:
        group.append(batch)
        if len(group) == accumulation_steps:
            yield group
            group = []
