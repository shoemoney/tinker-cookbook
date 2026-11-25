"""Load the generated intent-command curriculum and render it for supervised training."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import chz
import tinker
from pydantic import BaseModel, ConfigDict

from tinker_cookbook import renderers
from tinker_cookbook.recipes.trained_tools.commands import command_name
from tinker_cookbook.recipes.trained_tools.prompts import SYSTEM
from tinker_cookbook.renderers.tml_v0 import TmlV0Renderer
from tinker_cookbook.supervised.common import datum_from_model_input_weights
from tinker_cookbook.supervised.types import SupervisedDataset, SupervisedDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

MODEL_NAME = "thinkingmachines/Inkling-2-Mini"


class Function(BaseModel):
    name: str
    arguments: dict[str, str]


class Call(BaseModel):
    id: str
    type: Literal["function"]
    function: Function


class RawMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_calls: list[Call] = []
    tool_call_id: str | None = None
    name: str | None = None


class Sample(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    group: int
    skill: str
    phrase_family: str
    commands: list[str]
    results: list[str]
    answer: str
    messages: list[RawMessage]

    def request(self) -> str:
        return self.messages[1].content

    def conversation(self) -> list[renderers.Message]:
        messages: list[renderers.Message] = []
        for raw in self.messages:
            message = renderers.Message(role=raw.role, content=raw.content)
            if raw.tool_calls:
                message["tool_calls"] = [
                    renderers.ToolCall(
                        id=call.id,
                        function=renderers.ToolCall.FunctionBody(
                            name=call.function.name,
                            arguments=json.dumps(call.function.arguments, sort_keys=True),
                        ),
                    )
                    for call in raw.tool_calls
                ]
            if raw.tool_call_id is not None:
                message["tool_call_id"] = raw.tool_call_id
            if raw.name is not None:
                message["name"] = raw.name
            messages.append(message)
        return messages


@dataclass
class Task:
    """One request and the command names a correct run uses, in order ([] = answer directly)."""

    id: str
    request: str
    expected: list[str]

    @classmethod
    def from_sample(cls, sample: Sample) -> Task:
        """Expect only the commands issued before any follow-up user turn, which the evaluator never sends."""
        expected: list[str] = []
        for message in sample.messages[2:]:
            if message.role == "user":
                break
            expected += [
                command_name(call.function.arguments["command"]) for call in message.tool_calls
            ]
        return cls(sample.id, sample.request(), expected)


def load_samples(path: Path) -> list[Sample]:
    """Read one split and check that each conversation matches its recorded tool executions."""
    samples = [Sample.model_validate_json(line) for line in path.read_text().splitlines() if line]
    if not samples or len({row.id for row in samples}) != len(samples):
        raise ValueError(f"{path}: empty data or duplicate IDs")
    for row in samples:
        messages = row.conversation()
        if messages[0]["role"] != "system" or messages[0]["content"] != SYSTEM:
            raise ValueError(f"{row.id}: unexpected system prompt")
        if messages[-1]["role"] != "assistant" or messages[-1]["content"] != row.answer:
            raise ValueError(f"{row.id}: final turn differs from the recorded answer")
        calls = [call for m in messages for call in m.get("tool_calls", [])]
        results = [m["content"] for m in messages if m["role"] == "tool"]
        if len(calls) != len(row.commands) or results != row.results:
            raise ValueError(f"{row.id}: conversation differs from the recorded executions")
        for call, command in zip(calls, row.commands, strict=True):
            if call.function.name != "bash" or json.loads(call.function.arguments) != {
                "command": command
            }:
                raise ValueError(f"{row.id}: wrong command transport")
    return samples


def load_tasks(path: Path) -> list[Task]:
    """Tasks from a generated split, or from the blind test (rows with command names, no conversation)."""
    if "messages" not in json.loads(path.read_text().splitlines()[0]):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        return [Task(r["id"], r["request"], r["commands"]) for r in rows]
    return [Task.from_sample(s) for s in load_samples(path)]


def load_splits(data_path: str) -> dict[str, list[Sample]]:
    """Load train, valid and test; refuse any request that appears in two splits."""
    splits = {s: load_samples(Path(data_path) / f"{s}.jsonl") for s in ("train", "valid", "test")}
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        if {r.request() for r in splits[left]} & {r.request() for r in splits[right]}:
            raise ValueError(f"Splits share a request: {left}, {right}")
    return splits


def generation_prompt(
    renderer: renderers.Renderer, messages: list[renderers.Message], effort: float
) -> tinker.ModelInput:
    """Inkling renderers take a thinking effort; other renderers have no such setting."""
    if isinstance(renderer, TmlV0Renderer):
        return renderer.build_generation_prompt(messages, effort=effort)
    return renderer.build_generation_prompt(messages)


def render_sample(
    sample: Sample, renderer: renderers.Renderer, max_length: int, effort: float
) -> tinker.Datum:
    """Train on every assistant turn (calls and final answers); refuse to truncate."""
    train_on = renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES
    if isinstance(renderer, TmlV0Renderer):
        model_input, weights = renderer.build_supervised_example(
            sample.conversation(), train_on, effort=effort
        )
    else:
        model_input, weights = renderer.build_supervised_example(sample.conversation(), train_on)
    if model_input.length > max_length:
        raise ValueError(f"{sample.id}: {model_input.length} tokens exceed max_length={max_length}")
    return datum_from_model_input_weights(model_input, weights, reduction="mean")


class ToolsDataset(SupervisedDataset):
    def __init__(self, datums: list[tinker.Datum], batch_size: int, seed: int):
        self.datums = datums
        self.batch_size = batch_size
        self.seed = seed
        self.order = list(range(len(datums)))
        self.set_epoch()

    def __len__(self) -> int:
        return math.ceil(len(self.datums) / self.batch_size)

    def get_batch(self, index: int) -> list[tinker.Datum]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        batch = self.order[index * self.batch_size : (index + 1) * self.batch_size]
        return [self.datums[i] for i in batch]

    def set_epoch(self, seed: int = 0) -> None:
        self.order = list(range(len(self.datums)))
        random.Random(self.seed + seed).shuffle(self.order)


@chz.chz
class ToolsDatasetBuilder(SupervisedDatasetBuilder):
    data_path: str
    model_name: str = MODEL_NAME
    renderer_name: str
    batch_size: int = 32
    max_length: int = 4096
    effort: float = 0.0
    seed: int = 0

    def __call__(self) -> tuple[ToolsDataset, ToolsDataset]:
        renderer = renderers.get_renderer(self.renderer_name, get_tokenizer(self.model_name))
        splits = load_splits(self.data_path)
        train, valid = (
            ToolsDataset(
                [render_sample(r, renderer, self.max_length, self.effort) for r in splits[s]],
                self.batch_size,
                self.seed,
            )
            for s in ("train", "valid")
        )
        return train, valid
