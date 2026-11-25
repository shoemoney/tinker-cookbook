"""Closed-loop evaluation of the adapter, native tool declaration and a cheat sheet.

Every arm sees the same request and executes its own predicted commands against the fake tools
in ``tools.sh``. Compare a saved adapter with the stock model:

    python -m tinker_cookbook.recipes.trained_tools.eval \\
        data_path=~/tinker-runs/trained_tools/data model_path=tinker://.../sampler_weights/final
"""

from __future__ import annotations

import asyncio
import json
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import chz
import numpy as np
import tinker

from tinker_cookbook import cli_utils, model_info, renderers
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.recipes.trained_tools.commands import TOOLS, command_name, execute
from tinker_cookbook.recipes.trained_tools.data import (
    MODEL_NAME,
    Task,
    generation_prompt,
    load_tasks,
)
from tinker_cookbook.recipes.trained_tools.prompts import ARMS, arm_messages
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils.cost_estimation import TokenUsage, fetch_model_pricing

MAX_TURNS = 4
MAX_PROMPT_TOKENS = 4096


@dataclass
class Episode:
    task: Task
    arm: str
    commands: list[str] = field(default_factory=list)
    kinds: list[str] = field(default_factory=list)  # per turn: call, final, malformed, ...
    attempts: int = 0  # turns that tried to call a tool, parsed or not
    prompt_tokens: int = 0
    completion_tokens: int = 0
    answered: bool = False
    end: str = "turns_exhausted"  # answered, prompt_over_limit, malformed, parallel, invalid, ...
    messages: list[renderers.Message] = field(default_factory=list)


def attempted_call(message: renderers.Message) -> bool:
    """Whether a turn tried to call a tool, even if it did not parse (Inkling call marker)."""
    raw = message["content"] if isinstance(message["content"], str) else ""
    return bool(
        message.get("tool_calls")
        or message.get("unparsed_tool_calls")
        or "content_invoke_tool" in raw
    )


def parse_turn(
    message: renderers.Message, clean: bool, truncated: bool = False
) -> tuple[str, str | None]:
    """Classify one assistant turn as ``(kind, command)``.

    ``call`` carries the command line (a native call to one of the eight commands is rewritten
    into one); ``final`` is a plain answer; ``malformed`` is output the renderer could not parse;
    ``truncated`` hit the token limit; ``parallel`` is several calls in one turn, which the loop does
    not run; ``invalid`` is any other parsed call outside the allowed shapes.
    """
    if truncated:
        return "truncated", None
    if not clean or message.get("unparsed_tool_calls"):
        return "malformed", None
    calls = message.get("tool_calls", [])
    if not calls:
        return "final", None
    if len(calls) != 1:
        return "parallel", None
    try:
        name = calls[0].function.name.removeprefix("functions.")
        args = json.loads(calls[0].function.arguments)
    except json.JSONDecodeError:
        return "malformed", None
    if not isinstance(args, dict):
        return "invalid", None
    if name in TOOLS and len(args) == 1 and isinstance(text := next(iter(args.values())), str):
        return ("call", shlex.join([name, text])) if text else ("invalid", None)
    if name == "bash" and set(args) == {"command"} and isinstance(args["command"], str):
        return "call", args["command"]
    return "invalid", None


async def run_episode(
    task: Task,
    arm: str,
    client: tinker.SamplingClient,
    renderer: renderers.Renderer,
    effort: float,
    max_tokens: int,
) -> Episode:
    episode = Episode(task, arm, messages=arm_messages(task.request, arm, renderer))
    with TemporaryDirectory(prefix="tools-eval-") as tmp:
        for _ in range(MAX_TURNS):
            prompt = generation_prompt(renderer, episode.messages, effort)
            if prompt.length > MAX_PROMPT_TOKENS:
                episode.end = "prompt_over_limit"
                break
            response = await client.sample(
                prompt=prompt,
                num_samples=1,
                sampling_params=tinker.SamplingParams(
                    max_tokens=max_tokens, temperature=1.0, stop=renderer.get_stop_sequences()
                ),
            )
            tokens = response.sequences[0].tokens
            message, termination = renderer.parse_response(tokens)
            truncated = not termination.is_clean and len(tokens) >= max_tokens
            kind, command = parse_turn(message, termination.is_clean, truncated)
            episode.attempts += kind in ("call", "invalid", "parallel") or (
                kind in ("malformed", "truncated") and attempted_call(message)
            )
            episode.prompt_tokens += prompt.length
            episode.completion_tokens += len(tokens)
            episode.kinds.append(kind)
            episode.messages.append(message)
            if command is None:
                episode.answered = kind == "final"
                episode.end = "answered" if episode.answered else kind
                break
            episode.commands.append(command)
            try:
                result = await asyncio.to_thread(execute, command, Path(tmp) / "memory")
            except (ValueError, OSError, subprocess.SubprocessError):
                result = "ERROR: invalid command or argument"
            call = message.get("tool_calls", [])[0]
            call_id = call.id or f"call_{len(episode.commands)}"
            message["tool_calls"] = [call.model_copy(update={"id": call_id})]
            episode.messages.append(
                renderers.Message(
                    role="tool", name=call.function.name, tool_call_id=call_id, content=result
                )
            )
    return episode


def score(episode: Episode) -> dict[str, float]:
    """Per-episode counts; ``summarize`` sums them."""
    expected = episode.task.expected
    names = [command_name(c) for c in episode.commands]
    matched = 0
    for name in names:
        if matched < len(expected) and name == expected[matched]:
            matched += 1
    needed = bool(expected)
    return {
        "tool_needed": float(needed),
        "first_command_correct": float(needed and bool(names) and names[0] == expected[0]),
        "first_turn_malformed": float(needed and episode.kinds[:1] == ["malformed"]),
        "call_attempts": float(episode.attempts),
        "parsed_calls": float(sum(kind == "call" for kind in episode.kinds)),
        "invented_names": float(sum(name not in TOOLS for name in names)),
        "parallel_episodes": float("parallel" in episode.kinds),
        "exact_sequence": float(needed and episode.answered and names == expected),
        "extra_calls": float(len(names) - matched),
        "no_tool_needed": float(not needed),
        "no_tool_correct": float(not needed and episode.answered and not names),
        "prompt_tokens": float(episode.prompt_tokens),
        "completion_tokens": float(episode.completion_tokens),
    }


def summarize(episodes: list[Episode]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for episode in episodes:
        for key, value in score(episode).items():
            totals[key] = totals.get(key, 0.0) + value
    n = len(episodes)

    def ratio(num: str, den: str) -> float:
        return totals[num] / max(1.0, totals[den])

    return {
        "first_command_accuracy": ratio("first_command_correct", "tool_needed"),
        "first_turn_malformed": totals["first_turn_malformed"],
        "parsed_calls": totals["parsed_calls"],
        "call_attempts": totals["call_attempts"],
        "invented_names": totals["invented_names"],
        "parallel_episodes": totals["parallel_episodes"],
        "exact_sequence_accuracy": ratio("exact_sequence", "tool_needed"),
        "extra_calls_per_episode": totals["extra_calls"] / n,
        "no_tool_specificity": ratio("no_tool_correct", "no_tool_needed"),
        "prompt_tokens_per_episode": totals["prompt_tokens"] / n,
        "completion_tokens_per_episode": totals["completion_tokens"] / n,
        "episodes": float(n),
    }


def paired_bootstrap(
    a: list[float], b: list[float], draws: int = 10_000, seed: int = 0
) -> dict[str, float]:
    """Mean of ``a`` minus mean of ``b`` in points, with a 95% interval from episode-paired resampling."""
    if len(a) != len(b) or not a:
        raise ValueError("need two equal-length, non-empty per-episode score lists")
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    picks = np.random.default_rng(seed).integers(0, len(x), size=(draws, len(x)))
    diffs = 100 * (x[picks].mean(axis=1) - y[picks].mean(axis=1))
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {"diff": float(100 * (x.mean() - y.mean())), "low": float(low), "high": float(high)}


def first_command_scores(episodes: list[Episode]) -> list[float]:
    """Per tool-needed episode, whether the first command was right (paired by task order)."""
    return [score(e)["first_command_correct"] for e in episodes if e.task.expected]


class ToolsEvaluator(SamplingClientEvaluator):
    def __init__(
        self,
        tasks: list[Task],
        arm: str = "adapter",
        model_name: str = MODEL_NAME,
        effort: float = 0.0,
        max_tokens: int = 512,
        log_path: str | None = None,
        prefix: str = "valid/tools",
    ):
        if arm not in ARMS:
            raise ValueError(f"Unknown arm {arm!r}; expected one of {ARMS}")
        self.tasks, self.arm, self.effort = tasks, arm, effort
        self.episodes: list[Episode] = []
        self.max_tokens, self.log_path, self.prefix = max_tokens, log_path, prefix
        self.renderer = renderers.get_renderer(
            model_info.get_recommended_renderer_name(model_name), get_tokenizer(model_name)
        )

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        episodes = await asyncio.gather(
            *(
                run_episode(
                    task, self.arm, sampling_client, self.renderer, self.effort, self.max_tokens
                )
                for task in self.tasks
            )
        )
        self.episodes = list(episodes)
        if self.log_path:
            path = Path(self.log_path)
            path.mkdir(parents=True, exist_ok=True)
            rows = "".join(
                json.dumps(
                    {
                        "id": e.task.id,
                        "arm": e.arm,
                        "expected": e.task.expected,
                        "commands": e.commands,
                        "kinds": e.kinds,
                        "end": e.end,
                        "scores": score(e),
                        "messages": e.messages,
                    },
                    default=lambda x: x.model_dump(),
                )
                + "\n"
                for e in episodes
            )
            name = f"{self.prefix.replace('/', '_')}-{self.arm}-{uuid4().hex[:8]}.jsonl"
            (path / name).write_text(rows)
        return {f"{self.prefix}/{self.arm}/{k}": v for k, v in summarize(list(episodes)).items()}


@chz.chz
class ToolsEvaluatorBuilder:
    data_file: str
    arm: str = "adapter"
    model_name: str = MODEL_NAME
    effort: float = 0.0
    max_tokens: int = 512
    prefix: str = "valid/tools"
    log_path: str | None = None

    def __call__(self) -> ToolsEvaluator:
        tasks = load_tasks(Path(self.data_file))
        return ToolsEvaluator(
            tasks,
            self.arm,
            model_name=self.model_name,
            effort=self.effort,
            max_tokens=self.max_tokens,
            log_path=self.log_path,
            prefix=self.prefix,
        )


@chz.chz
class EvalConfig:
    """Compare the adapter with stock native tools and a cheat sheet on ``test.jsonl``."""

    data_path: str
    model_path: str
    model_name: str = MODEL_NAME
    test_file: str | None = None  # default data_path/test.jsonl; blind_test.jsonl also works
    arms: str = ",".join(ARMS)  # comma-separated subset of: adapter, native, sheet
    effort: float = 0.0
    max_tokens: int = 512
    log_path: str = cli_utils.runs_path("trained_tools", "comparison")
    base_url: str | None = None


async def main(cfg: EvalConfig) -> None:
    test_file = Path(cfg.test_file or Path(cfg.data_path) / "test.jsonl")
    name = test_file.stem
    tasks = load_tasks(test_file)
    service = tinker.ServiceClient(base_url=cfg.base_url)
    adapter, stock = await asyncio.gather(
        service.create_sampling_client(base_model=cfg.model_name, model_path=cfg.model_path),
        service.create_sampling_client(base_model=cfg.model_name),
    )
    arms = cfg.arms.split(",")
    evaluators = [
        ToolsEvaluator(
            tasks,
            arm=arm,
            model_name=cfg.model_name,
            effort=cfg.effort,
            max_tokens=cfg.max_tokens,
            log_path=cfg.log_path,
            prefix=f"{name}/tools",
        )
        for arm in arms
    ]
    results = await asyncio.gather(
        *(e(adapter if e.arm == "adapter" else stock) for e in evaluators)
    )
    rates = fetch_model_pricing(cfg.model_name)
    for arm, result in zip(arms, results, strict=True):
        prompt, completion = (
            result[f"{name}/tools/{arm}/{k}_tokens_per_episode"] * len(tasks)
            for k in ("prompt", "completion")
        )
        usage = TokenUsage(prefill_uncached=int(prompt), sampled=int(completion))
        result["list_price_usd"] = rates.estimate_usd(usage) if rates else float("nan")
    summary: dict[str, object] = dict(zip(arms, results, strict=True))
    if {"adapter", "native"} <= set(arms):
        by_arm = {e.arm: first_command_scores(e.episodes) for e in evaluators}
        summary["first_command_adapter_minus_native"] = paired_bootstrap(
            by_arm["adapter"], by_arm["native"]
        )
    Path(cfg.log_path).mkdir(parents=True, exist_ok=True)
    Path(cfg.log_path, f"comparison-{name}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main(chz.entrypoint(EvalConfig)))
