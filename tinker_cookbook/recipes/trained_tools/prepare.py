"""Generate the CC0 curriculum: every command runs in a fresh sandbox, and answers that need a
model are sampled once from the stock model.

    python -m tinker_cookbook.recipes.trained_tools.prepare --output ~/tinker-runs/trained_tools/data

Each split mixes single commands and two-step chains (independent sentence shapes), help-first and
unsupported-flag requests, follow-ups about a shown result, and many kinds of request that need no
command: definitions, translation, rewording, small talk, opinions, explanations, brainstorming,
trivia, mental arithmetic, word games, and questions about the tools themselves. Near-miss pairs
sit side by side, such as ``What is 7 + 8?`` and a long exact product, or ``What is a URL?`` and a
request to open one.
"""

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import shlex
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

import tinker

from tinker_cookbook import model_info, renderers
from tinker_cookbook.recipes.trained_tools import phrasings as P
from tinker_cookbook.recipes.trained_tools.commands import TOOLS, execute
from tinker_cookbook.recipes.trained_tools.data import MODEL_NAME, Sample, generation_prompt
from tinker_cookbook.recipes.trained_tools.overlap import Blocker
from tinker_cookbook.recipes.trained_tools.prompts import SYSTEM
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils.cost_estimation import TokenUsage, fetch_model_pricing

SAMPLED = (
    "define",
    "translate",
    "reword",
    "smalltalk",
    "opinion",
    "explain",
    "concept",
    "brainstorm",
    "factual",
)
MIX = {  # fraction of a split; "single" absorbs the rounding
    "single": 0.40,
    "chain": 0.10,
    "help": 0.05,
    "unsupported": 0.03,
    "follow_tool": 0.012,
    "follow_none": 0.048,
    "arith": 0.05,
    "wordgame": 0.05,
    "define": 0.04,
    "translate": 0.03,
    "reword": 0.03,
    "smalltalk": 0.025,
    "opinion": 0.025,
    "explain": 0.03,
    "concept": 0.05,
    "brainstorm": 0.025,
    "factual": 0.02,
}
SPLITS = {"train": (4000, 20261125), "valid": (400, 20261126), "test": (400, 20261127)}
FOLLOW_NONE = {
    "calc": [
        ("Is that even or odd?", "parity"),
        ("What is half of that?", "half"),
        ("How many digits does it have?", "digits"),
    ],
    "web_search": [
        ("Does that link use https?", "Yes."),
        ("Which domain is it on?", "example.org"),
    ],
    "image_search": [
        ("What file type is that link?", "PNG."),
        ("Is it a secure link?", "Yes, it uses https."),
    ],
    "image_gen": [
        ("Which folder was it saved in?", "/generated"),
        ("What file type is it?", "PNG."),
    ],
    "read_url": [
        ("What time does the archive open?", "09:00."),
        ("Does the page mention an opening time?", "Yes, 09:00."),
    ],
    "remember": [
        ("What exactly did you save?", "fact"),
        ("Did that get stored?", "Yes, it was saved."),
    ],
    "recall": [
        ("Was anything stored?", "No, nothing was found."),
        ("So there is no note on that?", "Right, nothing was found."),
    ],
    "ask_model": [
        ("Did the second model suggest anything?", "It suggested comparing alternatives.")
    ],
}
FOLLOW_TOOL = {  # command -> (follow-up request, second command, argument source)
    "calc": ("Save that number for later.", "remember", "result"),
    "web_search": ("Now open it.", "read_url", "result"),
    "image_search": ("Ask another model what caption suits it.", "ask_model", "result"),
    "read_url": ("Get a second model's view on that page.", "ask_model", "result"),
    "remember": ("Now check your notes for it.", "recall", "keyword"),
    "image_gen": ("Find an existing picture similar to that one.", "image_search", "scene"),
    "recall": ("Ask another model what to do about it.", "ask_model", "result"),
    "ask_model": ("Save that opinion in your notes.", "remember", "result"),
}
BLIND_TEST = Path(__file__).with_name("blind_test.jsonl")


@dataclass
class Follow:
    """A second user turn: a second command (``command``, argument from ``src``) or a direct ``answer``."""

    text: str
    command: str = ""
    src: str = ""
    answer: str = ""


@dataclass
class Row:
    kind: str
    skill: str
    family: str
    request: str
    steps: list[tuple[str, str]] = field(default_factory=list)
    answer: str | None = None
    chain: tuple[str, str] | None = None
    a: str = ""
    b: str = ""
    follow: Follow | None = None


def lower_first(text: str) -> str:
    return text if len(text) > 1 and text[1].isupper() else text[0].lower() + text[1:]


def expression(rng: random.Random) -> str:
    a, b, c = rng.randint(120, 9899), rng.randint(13, 999), rng.randint(7, 99)
    hi, lo = max(a, b), min(a, b)
    return rng.choice(
        [
            f"{a} * {b}",
            f"({a} + {b}) * {c}",
            f"{a} * {c} - {rng.randint(13, 839)}",
            f"({hi} - {lo}) * ({c} + 3)",
            f"{a * 4} / 4 + {b} * {c}",
            f"{a} + {b} * {c} - {rng.randint(100, 900)}",
        ]
    )


def small_sum(rng: random.Random) -> tuple[str, str]:
    kind, a, b = rng.randrange(4), rng.randint(2, 60), rng.randint(2, 60)
    if kind == 0:
        return f"{a} + {b}", str(a + b)
    if kind == 1:
        return f"{a + 40} - {b % 19 + 2}", str(a + 40 - (b % 19 + 2))
    if kind == 2:
        return f"{a % 11 + 2} * {b % 11 + 2}", str((a % 11 + 2) * (b % 11 + 2))
    return f"{(b % 8 + 2) * (a % 11 + 2)} / {b % 8 + 2}", str(a % 11 + 2)


class Generator:
    """One random stream per kind keeps the requests that need sampled answers reproducible."""

    def __init__(
        self,
        seed: int,
        used: set[str],
        answers: dict[str, str] | None,
        block: Blocker | None = None,
    ):
        self.seed, self.used, self.answers, self.block = seed, used, answers, block
        self.streams: dict[str, random.Random] = {}
        self.rng = random.Random(seed)

    def decorate(self, request: str) -> str:
        if self.rng.random() < 0.4:
            prefix = self.rng.choice(P.PREFIX)
            request = (prefix + lower_first(request) if prefix else request) + self.rng.choice(
                P.SUFFIX
            )
        return request

    def slot(self, name: str) -> str:
        pool = {
            "web_search": P.TOPICS,
            "image_search": P.SCENES,
            "image_gen": P.SCENES,
            "ask_model": P.QUESTIONS,
            "read_url": P.URLS,
            "remember": P.FACTS,
            "recall": P.RECALL_KEYS,
        }
        return expression(self.rng) if name == "calc" else self.rng.choice(pool[name])

    def single(self, name: str | None = None) -> Row:
        name = name or self.rng.choice(list(TOOLS))
        i = self.rng.randrange(len(P.SINGLE[name]))
        x = self.slot(name)
        request = self.decorate(P.SINGLE[name][i].format(x=x))
        return Row("single", name, f"single-{name}-{i}", request, [(name, x)])

    def chain(self) -> Row:
        pair = self.rng.choice(list(P.CHAIN))
        i = self.rng.randrange(len(P.CHAIN[pair]))
        n = self.rng.randrange(len(P.RECALL_KEYS))
        if pair == ("remember", "recall"):
            a, b = P.FACTS[n], P.RECALL_KEYS[n]
        else:
            first = {"calc": "calc", "read_url": "read_url", "image_search": "image_gen"}.get(
                pair[0], "web_search"
            )
            a, b = self.slot(first), ""
        request = self.decorate(P.CHAIN[pair][i].format(a=a, b=b))
        return Row(
            "chain", "+".join(pair), f"chain-{'-'.join(pair)}-{i}", request, chain=pair, a=a, b=b
        )

    def help(self) -> Row:
        row = self.single()
        (name, x), suffix = row.steps[0], self.rng.choice(P.HELP_SUFFIX)
        row.kind, row.skill = "help", "help"
        row.request = f"{row.request} {suffix}"
        row.steps = [(name, "-h"), (name, x)]
        return row

    def unsupported(self) -> Row:
        name, flag, i = (
            self.rng.choice(list(TOOLS)),
            self.rng.choice(P.FLAGS),
            self.rng.randrange(len(P.UNSUPPORTED)),
        )
        request = P.UNSUPPORTED[i].format(name=name, flag=flag, x=self.slot(name))
        return Row(
            "unsupported",
            "unsupported",
            f"unsupported-{i}",
            request,
            [(name, "-h")],
            answer=f"Unsupported flag: {flag}",
        )

    def follow(self, tool: bool) -> Row:
        name = self.rng.choice(list(FOLLOW_TOOL))
        row = self.single(name)
        row.kind, row.skill = "follow_tool" if tool else "follow_none", f"follow-{name}"
        if tool:
            text, command, src = FOLLOW_TOOL[name]
            row.follow = Follow(text, command=command, src=src)
        else:
            text, answer = self.rng.choice(FOLLOW_NONE[name])
            row.follow = Follow(text, answer=answer)
        return row

    def arith(self) -> Row:
        expr, answer = small_sum(self.rng)
        i = self.rng.randrange(len(P.ARITH_SHAPES))
        request = self.decorate(P.ARITH_SHAPES[i].format(e=expr))
        return Row("arith", "direct", f"arith-{i}", request, answer=answer)

    def wordgame(self) -> Row:
        game = self.rng.choice(list(P.WORDGAME))
        w, i = self.rng.choice(P.GAMEWORDS), self.rng.randrange(len(P.WORDGAME[game]))
        answer = {
            "rev": w[::-1],
            "up": w.upper(),
            "len": str(len(w)),
            "first": w[0].upper(),
            "last": w[-1].upper(),
            "vow": str(sum(c in "aeiou" for c in w)),
            "sort": "".join(sorted(w)),
        }[game]
        request = self.decorate(P.WORDGAME[game][i].format(w=w))
        return Row("wordgame", "direct", f"word-{game}-{i}", request, answer=answer)

    def phrase(self, kind: str) -> tuple[str, int]:
        """Draw one request of a kind that needs a sampled answer, with its shape index."""
        r = self.rng
        if kind == "define":
            i = r.randrange(len(P.DEFINE))
            return P.DEFINE[i].format(w=r.choice(P.WORDS)), i
        if kind == "translate":
            i = r.randrange(4)
            return P.TRANSLATE_SH[i].format(l=r.choice(P.LANGS), p=r.choice(P.TRANSLATE)), i
        if kind == "reword":
            i = r.randrange(4)
            return P.REWORD_SH[i].format(s=r.choice(P.STYLES), t=r.choice(P.REWORD)), i
        if kind == "explain":
            shape = r.choice(["Explain {}.", "Tell me {}.", "Could you describe {}?"])
            return shape.format(r.choice(P.EXPLAIN)), 0
        if kind == "concept":
            i = r.randrange(len(P.CONCEPT_SHAPES))
            return P.CONCEPT_SHAPES[i].format(c=r.choice(P.CONCEPTS)), i
        if kind == "brainstorm":
            i = r.randrange(4)
            return P.BRAIN_SH[i].format(b=r.choice(P.BRAIN)), i
        pool = {"smalltalk": P.SMALLTALK, "opinion": P.OPINIONS, "factual": P.FACTUAL}[kind]
        return r.choice(pool), 0

    def sampled(self, kind: str) -> Row:
        request, i = self.phrase(kind)
        request = self.decorate(request)
        answer = None if self.answers is None else self.answers.get(request)
        return Row(kind, "direct", f"{kind}-{i}", request, answer=answer)

    def make(self, kind: str) -> Row | None:
        self.rng = self.streams.setdefault(kind, random.Random(f"{self.seed}-{kind}"))
        draw = {
            "single": self.single,
            "chain": self.chain,
            "help": self.help,
            "unsupported": self.unsupported,
            "follow_tool": lambda: self.follow(True),
            "follow_none": lambda: self.follow(False),
            "arith": self.arith,
            "wordgame": self.wordgame,
        }.get(kind, lambda: self.sampled(kind))
        for _ in range(200):
            row = draw()
            texts = [row.request, row.follow.text if row.follow else ""]
            if self.block and any(t and self.block.blocked(t) for t in texts):
                continue
            if row.request not in self.used:
                self.used.add(row.request)
                return row
        return None

    def rows(self, n: int) -> list[Row]:
        quota = {k: round(v * n) for k, v in MIX.items()}
        quota["single"] += n - sum(quota.values())
        rows: list[Row] = []
        for kind, count in quota.items():
            sampled = kind in SAMPLED
            # With no answers yet, over-draw sampled kinds so later failures can be skipped.
            draws = int(count * 1.15) + 5 if sampled else count + 40
            got = 0
            for _ in range(draws):
                if got == count and not (sampled and self.answers is None):
                    break
                row = self.make(kind)
                if row is None or (sampled and self.answers is not None and row.answer is None):
                    continue
                rows.append(row)
                got += 1
            for _ in range(0 if self.answers is None else count - got):
                filler = self.make("single")  # a shortfall is filled with single commands
                if filler is None:
                    raise RuntimeError(f"could not fill the {kind} quota")
                rows.append(filler)
        random.Random(f"{self.seed}-shuffle").shuffle(rows)
        return rows


def number_in(text: str) -> float:
    """The numeric value of a calc result, or 0 when the text is not a finite number."""
    try:
        value = float(text)
    except ValueError:
        return 0.0
    return value if math.isfinite(value) else 0.0


def realize(row: Row, split: str, index: int, group: int) -> Sample:
    """Run the row's commands against the fake tools and build the conversation."""
    messages: list[dict[str, object]] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": row.request},
    ]
    commands: list[str] = []
    results: list[str] = []
    with TemporaryDirectory(prefix="intent-tools-") as tmp:

        def run(name: str, argument: str) -> str:
            command = shlex.join([name, argument])
            result = execute(command, Path(tmp) / "memory")
            call_id = f"call_{len(commands)}"
            call = {
                "id": call_id,
                "type": "function",
                "function": {"name": "bash", "arguments": {"command": command}},
            }
            messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
            messages.append(
                {"role": "tool", "name": "bash", "tool_call_id": call_id, "content": result}
            )
            commands.append(command)
            results.append(result)
            return result

        if row.chain:
            first, second = row.chain
            result = run(first, row.a)
            derived = {("remember", "recall"): row.b, ("image_gen", "image_search"): row.a}
            run(second, derived.get((first, second), result))
        else:
            for name, argument in row.steps:
                run(name, argument)
        answer = row.answer or (results[-1] if results else "")
        follow = row.follow
        if follow:
            messages.append({"role": "assistant", "content": results[-1]})
            messages.append({"role": "user", "content": follow.text})
            first_argument = row.steps[0][1]
            if follow.command:
                source = {
                    "result": results[-1],
                    "keyword": first_argument.split()[-1],
                    "scene": first_argument,
                }
                answer = run(follow.command, source[follow.src])
            else:
                number = number_in(results[0])
                answer = {
                    "parity": "Even." if int(number) % 2 == 0 else "Odd.",
                    "half": str(number / 2),
                    "digits": str(len(str(abs(int(number))))),
                    "fact": first_argument,
                }.get(follow.answer, follow.answer)
        messages.append({"role": "assistant", "content": answer})
    return Sample.model_validate(
        {
            "id": f"{split}-{index:05d}",
            "group": group + index,
            "skill": row.skill,
            "phrase_family": f"{split}-{row.family}",
            "commands": commands,
            "results": results,
            "answer": answer,
            "messages": messages,
        }
    )


def estimate_answers(requests: list[str]) -> dict[str, object]:
    """Upper bound for sampling the stock model's answers: three tries of 220 tokens each."""
    renderer = renderers.get_renderer(
        model_info.get_recommended_renderer_name(MODEL_NAME), get_tokenizer(MODEL_NAME)
    )
    prompts = sum(
        generation_prompt(
            renderer,
            [
                renderers.Message(role="system", content=SYSTEM),
                renderers.Message(role="user", content=r),
            ],
            0.0,
        ).length
        for r in requests
    )
    usage = TokenUsage(sampled=3 * 220 * len(requests), prefill_uncached=3 * prompts)
    rates = fetch_model_pricing(MODEL_NAME)
    return {
        "requests": len(requests),
        "upper_bound_usd": rates.estimate_usd(usage) if rates else None,
    }


async def sample_answers(
    requests: list[str], cache: Path, budget_usd: float | None = None
) -> dict[str, str]:
    """Ask the stock model once per request (effort 0.0, temperature 0.7, 220 tokens, 3 tries).

    Prints the upper-bound cost first and refuses if it exceeds ``budget_usd``. Each answer is
    appended to ``cache`` as it arrives, so an interrupted run keeps what it paid for.
    """
    answers = (
        {}
        if not cache.exists()
        else {r["request"]: r["answer"] for r in map(json.loads, cache.read_text().splitlines())}
    )
    todo = [r for r in requests if r not in answers]
    if not todo:
        return answers
    estimate = estimate_answers(todo)
    print(f"Sampling {len(todo)} answers; list-price upper bound ${estimate['upper_bound_usd']}")
    bound = estimate["upper_bound_usd"]
    if budget_usd is not None and (not isinstance(bound, float) or bound > budget_usd):
        raise SystemExit(f"Upper bound {bound} is not within --budget-usd {budget_usd}.")
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit(
            f"{len(todo)} answers need the stock model: set TINKER_API_KEY, "
            "or pass --answers with a cache that already holds them."
        )
    renderer = renderers.get_renderer(
        model_info.get_recommended_renderer_name(MODEL_NAME), get_tokenizer(MODEL_NAME)
    )
    client = await tinker.ServiceClient().create_sampling_client(base_model=MODEL_NAME)
    usage: list[tuple[int, int]] = []  # (prompt tokens, sampled tokens) per call
    cache.parent.mkdir(parents=True, exist_ok=True)

    with cache.open("a") as out:

        async def one(request: str) -> None:
            messages = [
                renderers.Message(role="system", content=SYSTEM),
                renderers.Message(role="user", content=request),
            ]
            for _ in range(3):
                prompt = generation_prompt(renderer, messages, 0.0)
                response = await client.sample(
                    prompt=prompt,
                    num_samples=1,
                    sampling_params=tinker.SamplingParams(
                        max_tokens=220, temperature=0.7, stop=renderer.get_stop_sequences()
                    ),
                )
                tokens = response.sequences[0].tokens
                message, termination = renderer.parse_response(tokens)
                text = renderers.get_text_content(message).strip()
                usage.append((prompt.length, len(tokens)))
                if termination.is_clean and not message.get("tool_calls") and 0 < len(text) <= 700:
                    answers[request] = text
                    out.write(json.dumps({"request": request, "answer": text}) + "\n")
                    out.flush()
                    return

        await asyncio.gather(*(one(r) for r in todo))
    rates = fetch_model_pricing(MODEL_NAME)
    billed = TokenUsage(prefill_uncached=sum(p for p, _ in usage), sampled=sum(c for _, c in usage))
    print(
        f"Sampled {len(todo)} answers; list price ${rates.estimate_usd(billed) if rates else 'unknown'}"
    )
    return answers


def build(
    output: Path,
    answers_file: Path | None = None,
    exclude: Path | None = BLIND_TEST,
    dry_run: bool = False,
    budget_usd: float | None = None,
) -> dict[str, object]:
    """Write train/valid/test JSONL and a manifest of hashes; samples answers when none are cached.

    Requests that overlap the ``exclude`` file (default: the shipped blind test) are never generated.
    """
    output.mkdir(parents=True, exist_ok=True)
    block = (
        Blocker([json.loads(line)["request"] for line in exclude.read_text().splitlines()])
        if exclude
        else None
    )
    requests: list[str] = []
    used: set[str] = set()
    for n, seed in SPLITS.values():
        requests += [
            r.request for r in Generator(seed, used, None, block).rows(n) if r.kind in SAMPLED
        ]
    if dry_run:
        return estimate_answers(requests)
    answers = asyncio.run(
        sample_answers(requests, answers_file or output / "answers.jsonl", budget_usd)
    )
    used = set()
    counts: dict[str, int] = {}
    kinds: dict[str, dict[str, int]] = {}
    files: dict[str, str] = {}
    for (split, (n, seed)), group in zip(SPLITS.items(), (100000, 500000, 600000), strict=True):
        rows = Generator(seed, used, answers, block).rows(n)
        samples = [realize(r, split, i, group) for i, r in enumerate(rows)]
        path = output / f"{split}.jsonl"
        path.write_text("".join(s.model_dump_json() + "\n" for s in samples))
        counts[split] = len(samples)
        kinds[split] = dict(Counter(r.kind for r in rows))
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest: dict[str, object] = {
        "license": "CC0-1.0",
        "counts": counts,
        "kinds": kinds,
        "files": files,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--answers", type=Path, help="cache of sampled answers (default: output/answers.jsonl)"
    )
    parser.add_argument(
        "--exclude",
        type=Path,
        default=BLIND_TEST,
        help="held-out JSONL of requests to stay clear of",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the answer-sampling estimate only"
    )
    parser.add_argument(
        "--budget-usd", type=float, help="refuse to sample if the upper-bound list price is higher"
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.output, args.answers, args.exclude, args.dry_run, args.budget_usd), indent=2
        )
    )
