"""LoRA SFT that teaches a model eight intent commands without tool descriptions in its prompt.

python -m tinker_cookbook.recipes.trained_tools.prepare --output ~/tinker-runs/trained_tools/data
python -m tinker_cookbook.recipes.trained_tools.train data_path=~/tinker-runs/trained_tools/data
"""

import asyncio
import json
import math
from datetime import datetime
from pathlib import Path

import chz

from tinker_cookbook import cli_utils, hyperparam_utils, model_info
from tinker_cookbook.recipes.trained_tools.data import (
    MODEL_NAME,
    ToolsDatasetBuilder,
    load_splits,
    render_sample,
)
from tinker_cookbook.recipes.trained_tools.eval import ToolsEvaluatorBuilder
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.supervised import train
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils.cost_estimation import TokenUsage, fetch_model_pricing


@chz.chz
class CLIConfig:
    data_path: str
    model_name: str = MODEL_NAME
    learning_rate: float | None = None  # default: hyperparam_utils.get_lr(model_name)
    batch_size: int = 32
    num_epochs: int = 1
    lora_rank: int = 32
    effort: float = 0.0  # Inkling thinking effort for training and evaluation
    max_length: int = 4096
    max_tokens: int = 512
    seed: int = 0
    eval_every: int = 50
    save_every: int = 0  # 0 saves only the final checkpoint; each save uses account storage
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    base_url: str | None = None
    dry_run: bool = False
    budget_usd: float | None = None  # refuse to start if training plus validation is higher
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


def preflight(cfg: CLIConfig) -> dict[str, object]:
    """Render every row locally and report sizes, overlap checks and the training estimate."""
    if not math.isfinite(cfg.effort) or not 0 <= cfg.effort < 1:
        raise ValueError("effort must be finite and in [0, 1)")
    splits = load_splits(cfg.data_path)  # raises if a request appears in two splits
    renderer = get_renderer(
        model_info.get_recommended_renderer_name(cfg.model_name), get_tokenizer(cfg.model_name)
    )
    tokens = {
        split: sum(
            render_sample(r, renderer, cfg.max_length, cfg.effort).model_input.length for r in rows
        )
        for split, rows in splits.items()
    }
    steps = math.ceil(len(splits["train"]) / cfg.batch_size) * cfg.num_epochs
    usage = TokenUsage(trained=tokens["train"] * cfg.num_epochs)
    # Each validation re-sends the growing conversation (about twice the tokens) and samples
    # the assistant turns once; the validation split's token count stands in for both.
    evals = math.ceil(steps / cfg.eval_every) if cfg.eval_every > 0 else 0
    validation = TokenUsage(
        prefill_uncached=2 * tokens["valid"] * evals, sampled=tokens["valid"] * evals
    )
    rates = fetch_model_pricing(cfg.model_name)
    usd = None if rates is None else rates.estimate_usd(usage)
    validation_usd = None if rates is None else rates.estimate_usd(validation)
    if usd is not None and validation_usd is not None:
        total = usd + validation_usd
        if cfg.budget_usd is not None and total > cfg.budget_usd:
            raise ValueError(
                f"Training ${usd:.2f} plus validation ${validation_usd:.2f} exceeds "
                f"budget_usd={cfg.budget_usd}"
            )
    return {
        "rows": {split: len(rows) for split, rows in splits.items()},
        "tokens_per_split": tokens,
        "steps": steps,
        "trained_tokens": usage.trained,
        "training_estimate_usd": usd,
        "validation_estimate_usd": validation_usd,
        "learning_rate": cfg.learning_rate or hyperparam_utils.get_lr(cfg.model_name),
    }


def build_config(cfg: CLIConfig, log_path: str) -> train.Config:
    renderer_name = model_info.get_recommended_renderer_name(cfg.model_name)
    return train.Config(
        log_path=log_path,
        model_name=cfg.model_name,
        recipe_name="recipe_trained_tools",
        renderer_name=renderer_name,
        dataset_builder=ToolsDatasetBuilder(
            data_path=cfg.data_path,
            model_name=cfg.model_name,
            renderer_name=renderer_name,
            batch_size=cfg.batch_size,
            max_length=cfg.max_length,
            effort=cfg.effort,
            seed=cfg.seed,
        ),
        learning_rate=cfg.learning_rate or hyperparam_utils.get_lr(cfg.model_name),
        lr_schedule="linear",
        num_epochs=cfg.num_epochs,
        lora_rank=cfg.lora_rank,
        evaluator_builders=[
            ToolsEvaluatorBuilder(
                data_file=str(Path(cfg.data_path) / "valid.jsonl"),
                model_name=cfg.model_name,
                effort=cfg.effort,
                max_tokens=cfg.max_tokens,
                log_path=log_path,
            )
        ],
        eval_every=cfg.eval_every,
        save_every=cfg.save_every,
        wandb_project=cfg.wandb_project,
        wandb_name=cfg.wandb_name,
        base_url=cfg.base_url,
        # Finish each update before validating so evaluation matches the saved weights.
        submit_ahead=0,
    )


async def cli_main(cfg: CLIConfig) -> None:
    report = preflight(cfg)
    print(json.dumps(report, indent=2))
    if cfg.dry_run:
        return
    run_name = cfg.wandb_name or f"{cfg.model_name.split('/')[-1]}-{datetime.now():%Y-%m-%d-%H-%M}"
    log_path = cfg.log_path or cli_utils.runs_path("trained_tools", run_name)
    cli_utils.check_log_dir(log_path, behavior_if_exists=cfg.behavior_if_log_dir_exists)
    await train.main(build_config(cfg, log_path))


if __name__ == "__main__":
    asyncio.run(cli_main(chz.entrypoint(CLIConfig)))
