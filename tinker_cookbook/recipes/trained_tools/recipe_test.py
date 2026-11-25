import asyncio
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import tinker

from tinker_cookbook import model_info, renderers
from tinker_cookbook.recipes.trained_tools import prepare, train
from tinker_cookbook.recipes.trained_tools.commands import TOOLS, command_name, execute
from tinker_cookbook.recipes.trained_tools.data import (
    MODEL_NAME,
    Task,
    generation_prompt,
    load_splits,
    load_tasks,
    render_sample,
)
from tinker_cookbook.recipes.trained_tools.eval import (
    Episode,
    ToolsEvaluator,
    paired_bootstrap,
    parse_turn,
    run_episode,
    score,
    summarize,
)
from tinker_cookbook.recipes.trained_tools.overlap import Blocker
from tinker_cookbook.recipes.trained_tools.prompts import ARMS, CHEAT_SHEET, arm_messages
from tinker_cookbook.renderers.tml_v0 import TmlV0Renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils.cost_estimation import TokenCostRates

RENDERER = cast(
    TmlV0Renderer,
    renderers.get_renderer(
        model_info.get_recommended_renderer_name(MODEL_NAME), get_tokenizer(MODEL_NAME)
    ),
)


SIZES = {"train": (400, 1), "valid": (60, 2), "test": (60, 3)}


def build_small(path: Path) -> dict[str, object]:
    """Small curriculum with stub answers for the requests that would need the model."""

    async def stub(
        requests: list[str], cache: Path, budget_usd: float | None = None
    ) -> dict[str, str]:
        return dict.fromkeys(requests, "A short answer.")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(prepare, "SPLITS", SIZES)
        patch.setattr(prepare, "sample_answers", stub)
        return prepare.build(path)


@pytest.fixture(scope="module")
def curriculum(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("curriculum")
    build_small(path)
    return path


def call(name: str, arguments: dict[str, str]) -> renderers.Message:
    body = renderers.ToolCall.FunctionBody(name=name, arguments=json.dumps(arguments))
    return renderers.Message(
        role="assistant", content="", tool_calls=[renderers.ToolCall(id="c1", function=body)]
    )


def test_tools_execute_and_stay_literal(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    for name in TOOLS:
        assert "Usage" in execute(f"{name} -h", memory)
    assert execute("calc '(1 + 2) * 3'", memory) == "9"
    execute("remember 'blue key'", memory)
    assert execute("recall 'blue'", memory) == "blue key"
    assert execute("recall 'blue'", tmp_path / "other") == "Not found"
    marker = tmp_path / "pwned"
    out = execute(f"remember '$(touch {marker}) `touch {marker}` > {marker}'", memory)
    assert "$(touch" in out and not marker.exists()
    for bad in [
        "rm -rf x",
        "calc",
        "calc 1 2",
        "calc --turbo 1",
        "web_search ''",
        "read_url ftp://x",
    ]:
        with pytest.raises((ValueError, subprocess.SubprocessError)):
            execute(bad, memory)
    assert command_name("calc '1+1'") == "calc" and command_name("'unterminated") == "INVALID"


def test_curriculum_is_deterministic_and_disjoint(curriculum: Path, tmp_path: Path) -> None:
    again = build_small(tmp_path)
    assert again["files"] == json.loads((curriculum / "manifest.json").read_text())["files"]
    splits = load_splits(str(curriculum))
    assert {len(rows) for rows in splits.values()} == {400, 60}
    requests = [r.request() for rows in splits.values() for r in rows]
    assert len(set(requests)) == len(requests)
    for name, source in (("train", "train"), ("valid", "train"), ("test", "test")):
        (tmp_path / f"{name}.jsonl").write_text((curriculum / f"{source}.jsonl").read_text())
    with pytest.raises(ValueError, match="share a request"):
        load_splits(str(tmp_path))


def test_follow_up_commands_are_not_expected_of_the_first_request(curriculum: Path) -> None:
    train = load_splits(str(curriculum))["train"]
    follow = [r for r in train if r.skill.startswith("follow") and len(r.commands) == 2]
    assert follow
    for sample in follow:
        assert Task.from_sample(sample).expected == [command_name(sample.commands[0])]
    help_first = next(r for r in train if r.skill == "help")  # both commands answer one request
    assert Task.from_sample(help_first).expected == [command_name(c) for c in help_first.commands]


def test_prepare_handles_negative_results_in_follow_ups() -> None:
    assert prepare.number_in("-7") == -7 and prepare.number_in("nan") == 0
    assert prepare.number_in("12.5") == 12.5 and prepare.number_in("oops") == 0


def test_paired_bootstrap() -> None:
    a, b = [1.0, 1.0, 0.0, 1.0] * 25, [0.0, 1.0, 0.0, 0.0] * 25
    result = paired_bootstrap(a, b)
    assert result["diff"] == 50 and result["low"] < 50 < result["high"]
    assert result == paired_bootstrap(a, b)  # seed 0: reproducible
    assert paired_bootstrap(a, a) == {"diff": 0.0, "low": 0.0, "high": 0.0}
    with pytest.raises(ValueError, match="equal-length"):
        paired_bootstrap([1.0], [1.0, 0.0])


def test_prepare_estimates_first_caps_spend_and_keeps_paid_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "answers.jsonl"
    monkeypatch.setattr(
        prepare, "fetch_model_pricing", lambda _: TokenCostRates(0.15, 0.015, 0.5, 0.75)
    )
    with pytest.raises(SystemExit, match="budget"):  # refuses before any sampling
        asyncio.run(prepare.sample_answers(["q1", "q2"], cache, budget_usd=1e-9))
    assert not cache.exists()

    class Flaky:
        calls = 0

        async def sample(self, *, prompt: tinker.ModelInput, **kwargs: object) -> SimpleNamespace:
            Flaky.calls += 1
            if Flaky.calls > 1:
                raise RuntimeError("service went away")
            conversation = [
                renderers.Message(role="system", content=prepare.SYSTEM),
                renderers.Message(role="user", content="q1"),
                renderers.Message(role="assistant", content="Fine."),
            ]
            target, _ = RENDERER.build_supervised_example(conversation, effort=0.0)
            tokens = target.to_ints()[prompt.length :]
            return SimpleNamespace(sequences=[SimpleNamespace(tokens=tokens)])

    class Service:
        async def create_sampling_client(self, **kwargs: object) -> Flaky:
            return Flaky()

    monkeypatch.setattr(prepare.tinker, "ServiceClient", Service)
    monkeypatch.setenv("TINKER_API_KEY", "test")
    with pytest.raises(RuntimeError, match="went away"):
        asyncio.run(prepare.sample_answers(["q1", "q2"], cache))
    assert [json.loads(line)["request"] for line in cache.read_text().splitlines()] == ["q1"]


def test_curriculum_mix_and_near_misses(curriculum: Path) -> None:
    train = load_splits(str(curriculum))["train"]
    assert {command_name(c) for r in train for c in r.commands} == set(TOOLS)
    no_command = [r for r in train if not r.commands]
    assert 0.3 < len(no_command) / len(train) < 0.5  # many kinds of request that need no command
    assert len({r.phrase_family for r in train}) > 100  # independent shapes, not one wrapper
    small = [
        r for r in no_command if r.skill == "direct" and r.answer.isdigit() and "+" in r.request()
    ]
    big = [r for r in train if r.skill == "calc"]
    assert small and big and all(not r.commands for r in small)
    assert any(
        len(r.commands) == 2 and "follow" in r.skill for r in train
    )  # follow-up needs a command
    assert any(r.skill.startswith("follow") and len(r.commands) == 1 for r in train)  # or does not
    assert {r.skill for r in train} >= {"help", "unsupported"}


def test_blind_test_is_shipped_and_kept_out_of_the_curriculum(curriculum: Path) -> None:
    blind = prepare.BLIND_TEST
    tasks = load_tasks(blind)
    assert len(tasks) == 120 and sum(bool(t.expected) for t in tasks) == 80
    assert {n for t in tasks for n in t.expected} == set(TOOLS)
    requests = [r.request() for rows in load_splits(str(curriculum)).values() for r in rows]
    assert not [r for r in requests if Blocker([t.request for t in tasks]).blocked(r)]


def test_only_assistant_turns_are_trained(curriculum: Path) -> None:
    sample = next(r for r in load_splits(str(curriculum))["train"] if len(r.commands) == 2)
    datum = render_sample(sample, RENDERER, 4096, 0.0)
    weights = datum.loss_fn_inputs["weights"].to_numpy()
    tokens = datum.model_input.to_ints()
    assert 0 < weights.sum() < len(weights)
    trained = RENDERER.tokenizer.decode([t for t, w in zip(tokens, weights, strict=True) if w > 0])
    assert sample.commands[0].split()[0] in trained and sample.messages[1].content not in trained
    with pytest.raises(ValueError, match="max_length"):
        render_sample(sample, RENDERER, 10, 0.0)


def test_arm_prompts() -> None:
    request = "Find a web page about cobalt gardens."
    tokens = {
        arm: generation_prompt(RENDERER, arm_messages(request, arm, RENDERER), 0.0).length
        for arm in ARMS
    }
    assert tokens == {"adapter": 58, "native": 909, "sheet": 313}
    adapter = arm_messages(request, "adapter", RENDERER)
    assert not any(name in adapter[0]["content"] for name in TOOLS)
    declared = json.loads(cast(str, arm_messages(request, "native", RENDERER)[1]["content"]))
    assert {t["function"]["name"] for t in declared} == set(TOOLS)
    sheet = arm_messages(request, "sheet", RENDERER)
    declared = json.loads(cast(str, sheet[1]["content"]))
    assert [t["function"]["name"] for t in declared] == ["bash"]  # declared, not pasted into text
    assert CHEAT_SHEET in str(sheet[0]["content"]) and "Available tool transport" not in str(sheet)
    with pytest.raises(ValueError, match="Unknown arm"):
        arm_messages(request, "bogus", RENDERER)


def test_parse_turn_and_scoring() -> None:
    assert parse_turn(call("bash", {"command": "calc '1+1'"}), True) == ("call", "calc '1+1'")
    assert parse_turn(call("functions.calc", {"text": "1+1"}), True) == ("call", "calc 1+1")
    assert parse_turn(renderers.Message(role="assistant", content="2"), True)[0] == "final"
    assert parse_turn(call("calc", {"x": "1", "y": "2"}), True)[0] == "invalid"
    assert parse_turn(call("rm", {"text": "x"}), True)[0] == "invalid"
    assert parse_turn(renderers.Message(role="assistant", content="x"), False)[0] == "malformed"
    assert (
        parse_turn(renderers.Message(role="assistant", content="x"), False, True)[0] == "truncated"
    )
    one = call("calc", {"text": "1"}).get("tool_calls", [])
    parallel = renderers.Message(role="assistant", content="", tool_calls=[*one, *one])
    assert parse_turn(parallel, True)[0] == "parallel"

    def episode(expected: list[str], commands: list[str], kinds: list[str], answered: bool):
        attempts = sum(kind != "final" for kind in kinds)
        return Episode(
            Task("x", "r", expected), "adapter", commands, kinds, attempts, 10, 5, answered
        )

    good = episode(["calc"], ["calc 1"], ["call", "final"], True)
    extra = episode(["calc"], ["calc 1", "calc 2"], ["call", "call", "final"], True)
    garbled = episode(["calc"], [], ["malformed"], False)
    direct = episode([], [], ["final"], True)
    invented = episode([], ["translate x"], ["call", "final"], True)
    assert score(good)["exact_sequence"] == 1 and score(extra)["extra_calls"] == 1
    assert (
        score(garbled)["first_turn_malformed"] == 1 and score(garbled)["first_command_correct"] == 0
    )
    assert score(direct)["no_tool_correct"] == 1 and score(invented)["no_tool_correct"] == 0
    several = episode(["calc"], [], ["parallel"], False)
    assert score(several)["parallel_episodes"] == 1 and score(good)["parallel_episodes"] == 0
    assert score(invented)["invented_names"] == 1 and score(good)["invented_names"] == 0
    summary = summarize([good, extra, garbled, direct, invented])
    assert summary["first_command_accuracy"] == pytest.approx(2 / 3)
    assert summary["no_tool_specificity"] == pytest.approx(1 / 2)
    assert (summary["parsed_calls"], summary["call_attempts"]) == (4, 5)
    assert summary["invented_names"] == 1


class Replay:
    """Replays rendered assistant turns after checking each prompt equals the expected one."""

    def __init__(self, conversation: list[renderers.Message], start: int):
        self.conversation, self.i = conversation, start

    async def sample(self, *, prompt: tinker.ModelInput, **kwargs: object) -> SimpleNamespace:
        target, _ = RENDERER.build_supervised_example(self.conversation[: self.i + 1], effort=0.0)
        expected = generation_prompt(RENDERER, self.conversation[: self.i], 0.0)
        assert prompt.to_ints() == expected.to_ints()
        self.i += 2
        return SimpleNamespace(
            sequences=[SimpleNamespace(tokens=target.to_ints()[prompt.length :])]
        )


@pytest.mark.parametrize("arm", ["adapter", "native"])
def test_closed_loop_executes_predicted_commands(arm: str, tmp_path: Path) -> None:
    task = Task("t", "Dig up a page on tides.", ["web_search"])
    result = execute("web_search tides", tmp_path / "m")
    conversation = arm_messages(task.request, arm, RENDERER)
    start = len(conversation)
    name, arguments = (
        ("web_search", {"text": "tides"})
        if arm == "native"
        else (
            "bash",
            {"command": "web_search tides"},
        )
    )
    conversation += [
        call(name, arguments),
        renderers.Message(role="tool", name=name, tool_call_id="c1", content=result),
        renderers.Message(role="assistant", content=result),
    ]
    client = cast(tinker.SamplingClient, Replay(conversation, start))
    episode = asyncio.run(run_episode(task, arm, client, RENDERER, 0.0, 512))
    assert episode.end == "answered" and episode.kinds == ["call", "final"]
    assert score(episode)["exact_sequence"] == 1


def test_garbled_call_json_is_malformed_and_logged(tmp_path: Path) -> None:
    garbled = (
        "<|message_model|><|content_thinking|>Need remember.<|end_message|><|message_model|>bash"
        '<|content_invoke_tool_json|>{"name"""bash","args":{"command":"remember x"}}<|end_message|>'
        "<|content_model_end_sampling|>"
    )
    tokens = RENDERER.tokenizer.encode(garbled, add_special_tokens=False)

    class Garbled:
        async def sample(self, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(sequences=[SimpleNamespace(tokens=tokens)])

    task = Task("g", "Jot down that x.", ["remember"])
    client = cast(tinker.SamplingClient, Garbled())
    metrics = asyncio.run(
        ToolsEvaluator([task], "native", log_path=str(tmp_path), prefix="x")(client)
    )
    assert metrics["x/native/first_turn_malformed"] == 1
    assert metrics["x/native/first_command_accuracy"] == 0
    rows = [
        json.loads(line) for p in tmp_path.glob("*.jsonl") for line in p.read_text().splitlines()
    ]
    assert len(rows) == 1 and rows[0]["end"] == "malformed" and rows[0]["commands"] == []
    with pytest.raises(ValueError, match="Unknown arm"):
        ToolsEvaluator([task], arm="stock")


def test_launcher_estimate_budget_and_forwarding(
    curriculum: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        train, "fetch_model_pricing", lambda _: TokenCostRates(0.15, 0.015, 0.5, 0.75)
    )
    cfg = train.CLIConfig(data_path=str(curriculum), lora_rank=8, eval_every=7, effort=0.5, seed=3)
    report = train.preflight(cfg)
    assert report["rows"] == {"train": 400, "valid": 60, "test": 60} and report["steps"] == 13
    assert report["training_estimate_usd"] == pytest.approx(
        cast(int, report["trained_tokens"]) * 0.75 / 1e6
    )
    valid_tokens = cast(dict[str, int], report["tokens_per_split"])["valid"]
    assert report["validation_estimate_usd"] == pytest.approx(  # two validations at steps 0 and 7
        2 * (2 * valid_tokens * 0.15 + valid_tokens * 0.5) / 1e6
    )
    with pytest.raises(ValueError, match="budget_usd"):
        train.preflight(train.CLIConfig(data_path=str(curriculum), budget_usd=0.0001))
    training_only = cast(float, report["training_estimate_usd"])
    with pytest.raises(ValueError, match="validation"):  # the cap covers validation too
        train.preflight(
            train.CLIConfig(data_path=str(curriculum), eval_every=7, budget_usd=training_only)
        )
    config = train.build_config(cfg, "/unused")
    assert (config.lora_rank, config.eval_every, config.submit_ahead) == (8, 7, 0)
    assert config.dataset_builder.effort == 0.5 and config.dataset_builder.seed == 3  # pyright: ignore[reportAttributeAccessIssue]
    monkeypatch.setattr(train, "fetch_model_pricing", lambda _: None)  # unlisted model: tokens only
    assert train.preflight(cfg)["training_estimate_usd"] is None
