# Trained tools: eight commands without tool descriptions

Fine-tune `thinkingmachines/Inkling-2-Mini` with LoRA SFT so that it calls eight one-line shell commands (`web_search`, `image_search`, `image_gen`, `ask_model`, `read_url`, `calc`, `remember`, `recall`) with **no tool text in its prompt**, and compare it with the stock model given the same commands in two ordinary ways. The point is an interface that is small and fixed enough to move into the weights; it does not teach a model to discover unknown tools.

## Setup and data

```bash
uv sync --extra dev
export DATA=~/tinker-runs/trained_tools/data
uv run python -m tinker_cookbook.recipes.trained_tools.prepare --output $DATA --dry-run   # estimate, no key
uv run python -m tinker_cookbook.recipes.trained_tools.prepare --output $DATA --budget-usd 0.5   # needs TINKER_API_KEY
```

`prepare` writes 4,000 train, 400 validation and 400 test conversations (CC0, generated here, no external data; it samples about 1,600 answers once from the stock model, roughly $0.15, prints the upper-bound cost before spending it, refuses above `--budget-usd`, and appends each answer to the cache as it arrives). Commands run through `tools.sh`, a deterministic fake backend: search returns `example.org` URLs, `calc` really computes, `remember` and `recall` use a per-conversation file, and each saved tool result is the command's actual output.

The curriculum is built to teach *when not to call* as well as how: single commands and two-step chains from many independent sentence shapes, help-first and unsupported-flag requests, follow-ups about a result already shown, and many kinds of request that need no command (definitions, translation, rewording, small talk, opinions, explanations, brainstorming, trivia, mental arithmetic, word games, questions about the tools themselves). Near-miss pairs sit side by side, such as `What is 7 + 8?` and a long exact product, or `What is a URL?` and a request to open one. An earlier version without those no-tool kinds invented commands for most unfamiliar questions.

`blind_test.jsonl` is a 120-request test (80 need a command, 40 do not), written by a separate author from the command descriptions alone and released under CC0 with the rest. `prepare` never generates a request that shares a phrase with it (`--exclude` changes the file). The file is **v1.1** of that test: it differs from the author's original (v1) only by a published substitution table that replaced real place names, local organisations, teams and retailers with invented ones and moved every domain under `.example`; no request was reworded and no label changed. No arm's results informed the table, which was committed before it was applied. No request is shared between splits; the generated test reuses the training templates and word pools, so it measures new combinations, not new phrasing, and the blind file is the test of unfamiliar phrasing.

## Train and evaluate

```bash
uv run python -m tinker_cookbook.recipes.trained_tools.train data_path=$DATA dry_run=True
uv run python -m tinker_cookbook.recipes.trained_tools.train data_path=$DATA wandb_name=mini-r32
# the final checkpoint is in ~/tinker-runs/trained_tools/mini-r32/checkpoints.jsonl
uv run python -m tinker_cookbook.recipes.trained_tools.eval data_path=$DATA model_path=tinker://<run>/sampler_weights/final
uv run python -m tinker_cookbook.recipes.trained_tools.eval data_path=$DATA model_path=tinker://<run>/sampler_weights/final \
    test_file=tinker_cookbook/recipes/trained_tools/blind_test.jsonl
```

Only the final checkpoint is saved by default (`save_every=0`); intermediate saves each use account checkpoint storage, and a full account stops training with a storage error. Defaults: LoRA rank 32, batch 32, one epoch (125 updates, about 457k trained tokens), learning rate from `hyperparam_utils.get_lr`, thinking effort 0.0, fresh optimizer, loss on every assistant turn. `train` prints a USD estimate from `utils/cost_estimation` for training and for the periodic validation, when the model is priced (`budget_usd` refuses a higher total). Validation runs before updates 0, 50 and 100 of 125, so the last in-loop point is step 100; `eval` on the test split is the check on the final weights. `eval` prints per-arm list-price cost from the tokens it used, and, when the adapter and native arms both ran, the paired bootstrap of their first-command difference (10,000 draws, seed 0; `arms=sheet` runs one arm alone).

Each arm gets the same request and runs its own predicted commands against the fake tools for up to four turns (temperature 1.0, 512 tokens per turn): **adapter** (the LoRA model, no tool text), **native** (stock model, the eight commands declared through the renderer's tool path, plus "Call one command per turn."), and **sheet** (stock model, a single declared `bash` tool and a ten-line cheat sheet of the commands in the system message). One parser serves all: a bash call with a one-line command, or a native call to one of the eight with one string argument. Several calls in one turn (`parallel`) or any other call shape end the episode and are counted. The adapter is trained with the short system prompt and evaluated with it plus one sentence about test fixtures, as are the other arms.

## Results

Two complete end-to-end runs of this recipe, from a clean checkout (`prepare` about $0.15, `train` about $0.34 plus $0.09 for the periodic validation, `eval` about $0.37 per run), about **$0.95** each at list prices. The generated test cells show **run 1 / run 2**; run 1 used the code before three small fixes (a stale deprecation, the checkpoint-storage default and the file name of the blind comparison) with the same training and scoring. Their test requests differ in 20 of 400 because the sampled answers differ. The adapter and native traces were rescored offline with the final scorer; **sheet** was re-sampled on each run's own requests with its `bash` tool declared through the renderer. The blind cells are two further samplings of the three arms on v1.1 with run 2's adapter (no retraining). One adapter per run, one seed, one sample per request, temperature 1.0.

Counts are per test; tokens are uncached prompt tokens summed over an episode's turns. *First command* is the first parsed call naming the expected command (a follow-up row expects only its first command, because the evaluator sends one user turn). *First turn malformed* is a first turn the renderer could not parse. *Parsed calls* are calls that parsed, of those attempted; a parsed call may still name a command that does not exist, which *invented names* counts across all arms. *Parallel* is episodes ended by several calls in one turn. *Exact sequence* needs the expected command names in order and a clean answer; *extra calls* are calls beyond an ordered match; *no-tool* is no-tool requests answered with no call.

**Generated test, 400 requests (250 need a command, 150 do not); run 1 / run 2**

| Arm | First command | First turn malformed | Parsed calls | Invented names | Parallel | Exact sequence | Extra calls | No-tool | Prompt tokens / episode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| adapter | 249 / 243 | 0 / 0 | 312/312, 311/311 | 2 / 0 | 0 / 0 | 248 / 243 | 3 / 9 | 148 / 142 | 167 / 167 |
| native | 238 / 238 | 0 / 0 | 360/362, 363/363 | 0 / 0 | 2 / 0 | 198 / 198 | 67 / 67 | 144 / 143 | 1,876 / 1,877 |
| sheet | 197 / 180 | 45 / 57 | 314/319, 306/314 | 7 / 1 | 0 / 0 | 153 / 137 | 69 / 78 | 144 / 143 | 709 / 705 |

**Blind test v1.1, 120 requests (80 / 40); sampling 1 / sampling 2**

| Arm | First command | First turn malformed | Parsed calls | Invented names | Parallel | Exact sequence | Extra calls | No-tool | Prompt tokens / episode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| adapter | 71 / 70 | 0 / 0 | 95/95, 92/92 | 2 / 4 | 0 / 0 | 67 / 65 | 11 / 10 | 36 / 35 | 176 / 172 |
| native | 77 / 78 | 0 / 0 | 156/158, 138/144 | 0 / 0 | 2 / 6 | 41 / 46 | 65 / 48 | 39 / 39 | 2,224 / 2,107 |
| sheet | 55 / 54 | 19 / 22 | 111/116, 111/114 | 6 / 5 | 2 / 2 | 28 / 31 | 45 / 49 | 39 / 40 | 764 / 740 |

- **The adapter is not better at choosing commands, and on unfamiliar phrasing it trails.** Adapter minus native on first command, paired bootstrap (10,000 draws, seed 0): +4.4 points [+1.6, +7.2] and +2.0 [-1.2, +5.6] on the generated test in runs 1 and 2, and -7.5 [-15.0, 0.0] and -10.0 [-18.8, -2.5] on the blind test. What it saves is the prompt (about 8 to 9% of native's uncached tokens, 167 against 1,877 on the generated test; the declaration is a cacheable prefix, so this overstates the saving on a service that caches it) and extra calls (3 to 11 against 48 to 67). Token counts are not latency figures.
- **It over-calls on casual no-tool requests and invents commands.** On the blind test the adapter made a call on 4 and 5 of 40 no-tool requests (`ask_model` for an explanation or an opinion, `calc` for word problems, `web_search`, and an invented `name_ideas`), against 1 for native. Its invented names were `current_mortgage_rate` in sampling 1; and `get_mortgage_rate_trend`, `name_ideas` and `open_url` in sampling 2; the other counted call in each is a `remember` whose argument had an unterminated quote. On the original v1 file the final run's adapter had invented `capital_lookup`, `get_latest_sports_score` and `get_current_mortgage_rate`; native never invents a name, because the renderer only offers eight. The generated test is built from the training templates, so its 142 to 148 of 150 says little about unfamiliar phrasing.
- **Native's blind misses are mostly parallel calls, not wrong commands.** Two to six of its 120 blind episodes ended on a turn with several calls, which the loop does not run; reading such a turn by its first call could add at most that many first commands to native's count.
- **Sheet mostly measures a plain-text call.** Its first turn is malformed in 19 to 57 episodes, and in 18 of 19, 21 of 22, 44 of 45 and 55 of 57 of them the visible text names the expected command first, written as `image_search "..."` rather than as a call. Declaring the `bash` tool through the renderer helped only a little (180 against 169 first commands in an earlier version of this arm), because the cheat sheet itself shows the plain-text form. Read the sheet rows as "cheat sheet in the system message", not as command choice.
- **Run-to-run spread is not small.** Two runs of identical code moved the adapter by 6 of 250 on first command (249 against 243), 6 of 150 on no-tool and 5 of 250 on exact sequence; extra calls went from 3 to 9; the 40-request blind no-tool row moved from 40/40 to 34/40 between earlier runs. Expect a few points of movement in every cell, more on the 40-request row. The study this recipe comes from (blind v1 file, one adapter) got 76/80 and 39/40 for the adapter.
- The fake backend does not test search relevance, image quality, URL safety or open-ended reasoning, and new commands need retraining.

## Tests

```bash
pytest tinker_cookbook/recipes/trained_tools
pyright tinker_cookbook/recipes/trained_tools
pre-commit run --files tinker_cookbook/recipes/trained_tools/*
```

No API key is needed. They cover command execution and quoting, deterministic splits with no shared request, the curriculum's mix and near-miss pairs, which commands a follow-up row expects, loss masking, the three prompts (the sheet's `bash` declared through the renderer), call parsing and scoring (invented names, parallel calls), the paired bootstrap, a closed loop against replayed model output (including garbled call JSON), the blind file's shape and its separation from the curriculum, `prepare`'s estimate, budget cap and incremental answer cache, and the launcher's training-plus-validation estimate and config forwarding.
