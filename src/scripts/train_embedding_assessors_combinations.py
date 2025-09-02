from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from src.utils.config import load_config
from src.visualisations.embedding_assessors import (
    plot_assessor_metrics,
    load_metrics_json,
)


@dataclass(frozen=True)
class ExperimentGrid:
    cfg_file: Path = Path("params.yaml")
    embedding_model_id: str = "openai/text-embedding-3-large"
    dataset_prompt_map: Dict[str, List[str]] = field(
        default_factory=dict, init=False
    )
    model_ids: Tuple[str, ...] = (
        "llama3.1_8b_chat",
        "mistral_7b_instruct",
        "ministral_8b_instruct",
        "llama3.3_70b",
        "qwen_2.5_7b_instruct",
        "deepseek_qwen_32b",
    )
    top_k_values: Tuple[int, ...] = (3072,)
    classifier_types: Tuple[str, ...] = ("xgboost", "logistic")

    def __post_init__(self):
        object.__setattr__(
            self,
            "dataset_prompt_map",
            {
                "birth_years_4k": ["base"],
                "cities_10k": ["base"],
                "gsm8k": ["base_3_shot", "cot_3_shot"],
                "math_operations_6k": ["base"],
                "medals_9k": ["base"],
                "notable_people": ["base"],
                "trivia_qa_2_60k": ["base"],
            },
        )
        object.__setattr__(
            self,
            "dataset_hold_out_map",
            {
                "trivia_qa_2_60k": 10000,
                "gsm8k": 0,
                "birth_years_4k": 0,
                "cities_10k": 0,
                "math_operations_6k": 0,
                "notable_people": 0,
                "medals_9k": 0,
            },
        )

    def default_prompt_for(self, dataset_id: str) -> str:
        return self.dataset_prompt_map[dataset_id][0]


GRID = ExperimentGrid()

logger = logging.getLogger("assessor-orchestrator")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)


def load_evaluations_dir(cfg_path: Path) -> Path:
    cfg = load_config(cfg_path)
    return Path(
        cfg.get("base", {}).get("evaluations_dir", "./evaluations")
    ).expanduser()


def run_subprocess(cmd: List[str]) -> None:
    logger.info("▶ %s", " ".join(cmd))
    with subprocess.Popen(
        cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    ) as proc:
        assert proc.stdout
        for line in proc.stdout:
            logger.info("%s", line.rstrip())
        if proc.wait() != 0:
            logger.error("Child process failed → abort")
            sys.exit(1)


def parse_experiment_name(exp_name: str) -> Optional[Tuple[str, str, str, int]]:
    if "_top_" not in exp_name:
        return None
    prefix, top_k_str = exp_name.rsplit("_top_", 1)
    try:
        top_k = int(top_k_str)
    except ValueError:
        return None
    for ds, prompts in GRID.dataset_prompt_map.items():
        for pv in prompts:
            token = f"_{ds}_{pv}"
            if prefix.endswith(token):
                return prefix[: -len(token)], ds, pv, top_k
    return None


def train_all_combinations(
    *,
    ds_filter: str | None,
    model_filter: str | None,
    cls_filter: str | None,
    topk_filter: int | None,
):
    logger.info("--- Training Phase ---")
    eval_dir = load_evaluations_dir(GRID.cfg_file)
    assessors_dir = eval_dir / "assessors"

    for classifier_type in GRID.classifier_types:
        if cls_filter and classifier_type != cls_filter:
            continue
        for model_id in GRID.model_ids:
            if model_filter and model_id != model_filter:
                continue
            for top_k in GRID.top_k_values:
                if topk_filter and top_k != topk_filter:
                    continue
                for (
                    train_ds,
                    prompt_versions,
                ) in GRID.dataset_prompt_map.items():
                    if ds_filter and train_ds != ds_filter:
                        continue
                    for prompt_version in prompt_versions:
                        test_datasets = [
                            ds
                            for ds in GRID.dataset_prompt_map
                            if ds != train_ds
                        ]
                        exp_name = f"{model_id}_{train_ds}_{prompt_version}_top_{top_k}"
                        pv_arg = ",".join(
                            f"{ds}:{(prompt_version if ds == train_ds else GRID.default_prompt_for(ds))}"
                            for ds in GRID.dataset_prompt_map
                        )
                        cmd = [
                            sys.executable,
                            "-m",
                            "src.stages.train_embedding_assessors",
                            "--config",
                            str(GRID.cfg_file),
                            "--embedding-model-id",
                            GRID.embedding_model_id,
                            "--classifier-type",
                            classifier_type,
                            "--prompt-versions",
                            pv_arg,
                            "--model-id",
                            model_id,
                            "--top-k",
                            str(top_k),
                            "--train-datasets",
                            train_ds,
                            "--test-datasets",
                            *test_datasets,
                            "--experiment-name",
                            exp_name,
                            "--random-state",
                            "42",
                            "--holdout-size",
                            str(GRID.dataset_hold_out_map.get(train_ds, 0)),
                        ]
                        run_subprocess(cmd)
                        metrics_file = (
                            assessors_dir
                            / exp_name
                            / f"{classifier_type}_metrics.json"
                        )
                        if not metrics_file.exists():
                            logger.warning("Metrics missing: %s", metrics_file)


def discover_metrics() -> Dict[str, Dict[Tuple[str, int], Dict[str, Path]]]:
    """Walk assessor folders and group metrics by classifier, (train_ds, top_k)."""

    logger.info("--- Discovery Phase (finding existing metrics) ---")
    eval_dir = load_evaluations_dir(GRID.cfg_file)
    assessors_dir = eval_dir / "assessors"
    output: Dict[str, Dict[Tuple[str, int], Dict[str, Path]]] = {
        ct: {} for ct in GRID.classifier_types
    }

    for classifier_type in GRID.classifier_types:
        count = 0
        for metrics_file in assessors_dir.rglob(
            f"{classifier_type}_metrics.json"
        ):
            parsed = parse_experiment_name(metrics_file.parent.name)
            if not parsed:
                logger.debug(
                    "Skipping folder with unknown naming: %s",
                    metrics_file.parent,
                )
                continue
            model_id, train_ds, prompt_version, top_k = parsed
            label_suffix = f"(top-{top_k}, {classifier_type})"
            label = (
                f"{model_id} ({prompt_version}, {label_suffix})"
                if prompt_version != "base"
                else f"{model_id} {label_suffix}"
            )
            key = (train_ds, top_k)
            output[classifier_type].setdefault(key, {})[label] = metrics_file
            count += 1
        logger.info("Discovered %d metrics for %s", count, classifier_type)
    return output


def build_summaries_and_plots(
    *,
    metrics_by_classifier: Dict[str, Dict[Tuple[str, int], Dict[str, Path]]],
    metric_name: str,
    which: str,
) -> None:

    out_dir = Path("./plots/assessors/")
    out_dir.mkdir(parents=True, exist_ok=True)

    for classifier_type, dataset_map in metrics_by_classifier.items():
        if not dataset_map:
            logger.warning(
                "No metrics for %s – skipping plotting", classifier_type
            )
            continue

        logger.info("--- Plotting for %s ---", classifier_type)
        for (train_ds, top_k), mapping in dataset_map.items():
            existing = {lbl: p for lbl, p in mapping.items() if p.exists()}
            if not existing:
                logger.warning(
                    "No existing files for %s (top %d, %s)",
                    train_ds,
                    top_k,
                    classifier_type,
                )
                continue

            logger.info("Processing %s | top_k=%d", train_ds, top_k)
            results: Dict[str, Dict[str, object]] = {}
            for label, path in existing.items():
                try:
                    agg_df = load_metrics_json(
                        str(path), which="aggregated", metric_name=metric_name
                    )
                    agg_val = float(agg_df["metric"].iloc[0])
                    by_df = load_metrics_json(
                        str(path), which="by_dataset", metric_name=metric_name
                    )
                    per_ds = {
                        row.test_dataset: float(row.metric)
                        for row in by_df.itertuples()
                        if hasattr(row, "test_dataset")
                    }
                    results[label] = {"agg": agg_val, "by_ds": per_ds}
                except Exception as e:
                    logger.error("Failed to load metrics %s: %s", path, e)

            if not results:
                logger.warning(
                    "No usable metrics for %s (top %d) – skip", train_ds, top_k
                )
                continue

            # Pick best checkpoint per base model by aggregated value
            best: Dict[str, str] = {}
            for label, stats in results.items():
                model_base = label.split()[0]
                if (
                    model_base not in best
                    or stats["agg"] > results[best[model_base]]["agg"]
                ):
                    best[model_base] = label

            summary = {
                "train_dataset": train_ds,
                "top_k": top_k,
                "classifier_type": classifier_type,
                "metric": metric_name,
                "models": {
                    mb: {
                        "best_label": lbl,
                        "aggregated": results[lbl]["agg"],
                        "by_dataset": results[lbl]["by_ds"],
                    }
                    for mb, lbl in best.items()
                },
            }
            summary_file = (
                out_dir
                / f"{train_ds}_top_{top_k}_{metric_name}_{classifier_type}_summary.json"
            )
            summary_file.write_text(
                json.dumps(summary, indent=2, ensure_ascii=False)
            )
            logger.info("Wrote summary → %s", summary_file)

            pdf_file = (
                out_dir
                / f"{train_ds}_top_{top_k}_{metric_name}_{classifier_type}.pdf"
            )
            try:
                plot_assessor_metrics(
                    metrics_json_paths=existing,
                    metric_name=metric_name,
                    which=which,
                    save_path=pdf_file,
                )
                logger.info("Plot saved → %s", pdf_file)
            except Exception as e:
                logger.error("Plot failed for %s: %s", pdf_file, e)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metric", default="auc_roc")
    p.add_argument(
        "--view", choices=["by_dataset", "aggregated"], default="by_dataset"
    )
    p.add_argument("--skip-training", action="store_true")
    p.add_argument("--train-dataset")
    p.add_argument("--model-id")
    p.add_argument("--classifier")
    p.add_argument("--top-k", type=int)
    args = p.parse_args()

    if not args.skip_training:
        train_all_combinations(
            ds_filter=args.train_dataset,
            model_filter=args.model_id,
            cls_filter=args.classifier,
            topk_filter=args.top_k,
        )

    metrics_map = discover_metrics()
    build_summaries_and_plots(
        metrics_by_classifier=metrics_map,
        metric_name=args.metric,
        which=args.view,
    )
    logger.info("--- Finished Embedding Assessor Training and Evaluation ---")


if __name__ == "__main__":
    main()
