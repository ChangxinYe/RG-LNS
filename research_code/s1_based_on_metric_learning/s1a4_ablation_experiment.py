"""
Run the cumulative RG-LNS ablation used by Table 5 with one command.

The driver evaluates Gallagher-initialized RG-LNS on ImageNet-LSEJ-10
(2-pixel erosion), GAP-5, and JPwLEG-5.  It deliberately delegates each
dataset/variant pair to the existing dataset evaluator in a separate Python
process.  This keeps the published evaluation implementations authoritative,
releases CUDA state between runs, and gives every run its own log and status.

The two executed variants are sufficient to recover all four cumulative
table rows:

1. reliability-guided destroy, fixed component position, B=1;
2. reliability-guided destroy, all component translations, B=4.

Reliable components always use reciprocal Top-K adjacencies supported by
complete 2x2 cycles.  This is one component-construction rule rather than two
independent ablation modules.

The final evaluator already reports the initial Gallagher baseline and its
strictly comparable B=1 reference, so separate baseline and B=1 runs would be
redundant.  This script writes summary.txt, results.json, and results.csv; it
does not generate LaTeX.  Running this file without arguments evaluates the
complete test splits using the one-click defaults below.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
MODULE_DIR = SCRIPT_PATH.parent
PROJECT_ROOT = MODULE_DIR.parent
DEFAULT_OUTPUT_ROOT = MODULE_DIR / "eval_result" / SCRIPT_PATH.stem

DEFAULT_LSEJ_DATA_ROOT = PROJECT_ROOT / "datasets" / "ImageNet_LSEJ" / "ImageNet_LSEJ"
DEFAULT_GAP_DATA_ROOT = PROJECT_ROOT / "datasets" / "GAP_fast"
DEFAULT_JPLEG_DATA_ROOT = PROJECT_ROOT / "datasets" / "MET_Dataset"

DEFAULT_LSEJ_CHECKPOINT = (
    MODULE_DIR
    / "logs"
    / "s1a_train_lsej"
    / "grid10_erode2"
    / "hard_triplet_d128_s224_k15_2026-07-25-10-22-41"
    / "checkpoints"
    / "best.pth"
)
DEFAULT_GAP_CHECKPOINT = (
    MODULE_DIR
    / "logs"
    / "s1a_train_gap"
    / "GAP-5"
    / "hard_triplet_d128_s224_k15_2026-07-31-20-43-26"
    / "checkpoints"
    / "best.pth"
)
DEFAULT_JPLEG_CHECKPOINT = (
    MODULE_DIR
    / "logs"
    / "s1a_train_jpleg"
    / "jpleg5"
    / "hard_triplet_d128_s224_2026-07-28-23-27-31"
    / "checkpoints"
    / "best.pth"
)

# One-click defaults.  Edit these values when launching directly from an IDE.
DEFAULT_DATASETS = ("lsej", "gap", "jpleg")
DEFAULT_DEVICE = "auto"
DEFAULT_GPU_ID = 0
DEFAULT_MAX_SAMPLES = 0
DEFAULT_SCORE_BATCH_SIZE = 128
DEFAULT_PROGRESS_INTERVAL = 25
DEFAULT_MUTUAL_TOP_K = 3
DEFAULT_MIN_COMPONENT_SIZE = 4
DEFAULT_COMPONENT_INDEX = 0
DEFAULT_REFINEMENT_ROUNDS = 1
DEFAULT_PARTIAL_COMPLETION_BEAM_WIDTH = 2

REINFORCED_COMPLETION_SOURCE_PATHS = tuple(
    sorted(
        (
            MODULE_DIR
            / "layout_refiners"
            / "reinforced_component_reassembly"
        ).glob("*.py")
    )
)

COMMON_RESULT_SOURCE_PATHS = (
    MODULE_DIR / "assembly_solvers" / "our_iterative_component_reassembly" / "single_component_e1.py",
    MODULE_DIR / "assembly_solvers" / "our_iterative_component_reassembly" / "profiled_completion_e1.py",
    *REINFORCED_COMPLETION_SOURCE_PATHS,
    MODULE_DIR / "metric_geometry.py",
    MODULE_DIR / "metric_lsej_macro_assisted.py",
    MODULE_DIR / "metric_scorer.py",
    MODULE_DIR / "metric_transforms.py",
    MODULE_DIR / "metric_vit_model.py",
    PROJECT_ROOT / "models" / "vit_handwritten.py",
    PROJECT_ROOT / "baselines" / "PuzzleDemoMGC_CVPR2012_Python" / "mgc.py",
)

DATASET_RESULT_SOURCE_PATHS = {
    "lsej": (
        MODULE_DIR / "metric_lsej_data.py",
        MODULE_DIR / "metric_lsej_evaluator.py",
        MODULE_DIR / "s1a_eval_lsej.py",
        MODULE_DIR / "s1a3_eval_lsej_iterative_component_reassembly.py",
    ),
    "gap": (
        MODULE_DIR / "metric_gap_data.py",
        MODULE_DIR / "metric_gap_evaluator.py",
        MODULE_DIR / "metric_jpleg_evaluator.py",
        MODULE_DIR / "s1a_eval_gap.py",
        MODULE_DIR / "s1a3_eval_gap_iterative_component_reassembly.py",
    ),
    "jpleg": (
        PROJECT_ROOT / "baselines" / "Edge2Vec_arxiv_2022" / "jpleg_data.py",
        MODULE_DIR / "metric_jpleg_data.py",
        MODULE_DIR / "metric_jpleg_evaluator.py",
        MODULE_DIR / "s1a_eval_jpleg.py",
        MODULE_DIR / "s1a3_eval_jpleg_iterative_component_reassembly.py",
    ),
}


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    module: str
    selector_argument: str
    selector_value: str
    full_sample_count: int
    expected_data_entry: str


@dataclass(frozen=True)
class VariantSpec:
    key: str
    label: str
    translation_mode: str
    completion_beam_width: int


@dataclass(frozen=True)
class TableRowSpec:
    key: str
    label: str
    run_variant: str
    metric_section: str


DATASETS: dict[str, DatasetSpec] = {
    "lsej": DatasetSpec(
        key="lsej",
        display_name="ImageNet-LSEJ-10 (2 px)",
        module=(
            "s1_based_on_metric_learning."
            "s1a4_eval_lsej_profiled_large_neighborhood_reassembly"
        ),
        selector_argument="--task",
        selector_value="grid10_erode2",
        full_sample_count=2000,
        expected_data_entry="configs",
    ),
    "gap": DatasetSpec(
        key="gap",
        display_name="GAP-5",
        module=(
            "s1_based_on_metric_learning."
            "s1a4_eval_gap_profiled_large_neighborhood_reassembly"
        ),
        selector_argument="--dataset",
        selector_value="GAP-5",
        full_sample_count=3000,
        expected_data_entry="GAP-5",
    ),
    "jpleg": DatasetSpec(
        key="jpleg",
        display_name="JPwLEG-5",
        module=(
            "s1_based_on_metric_learning."
            "s1a4_eval_jpleg_profiled_large_neighborhood_reassembly"
        ),
        selector_argument="--dataset",
        selector_value="jpleg5",
        full_sample_count=2000,
        expected_data_entry="JPLEG-5",
    ),
}

RUN_VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec(
        key="reliability_destroy_fixed_b1",
        label="Reliability-Guided Destroy",
        translation_mode="fixed",
        completion_beam_width=1,
    ),
    VariantSpec(
        key="full_rg_lns_b4",
        label="Full RG-LNS (B=4)",
        translation_mode="all",
        completion_beam_width=4,
    ),
)

TABLE_ROWS: tuple[TableRowSpec, ...] = (
    TableRowSpec(
        key="vit_t_gallagher",
        label="ViT-T + Gallagher",
        run_variant="full_rg_lns_b4",
        metric_section="baseline",
    ),
    TableRowSpec(
        key="reliability_guided_destroy",
        label="+ Reliability-Guided Destroy",
        run_variant="reliability_destroy_fixed_b1",
        metric_section="s1a4",
    ),
    TableRowSpec(
        key="component_translations",
        label="+ Component Translations",
        run_variant="full_rg_lns_b4",
        metric_section="s1a3_single_path_reference",
    ),
    TableRowSpec(
        key="beam_search_completion",
        label="+ Beam-Search Completion",
        run_variant="full_rg_lns_b4",
        metric_section="s1a4",
    ),
)


def _now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _path_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _source_fingerprint(path: Path) -> dict[str, Any]:
    payload = _path_fingerprint(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    payload["sha256"] = digest.hexdigest()
    return payload


def _signature(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _checkpoint_for(args: argparse.Namespace, dataset: str) -> Path:
    return Path(getattr(args, f"{dataset}_checkpoint")).expanduser().resolve()


def _data_root_for(args: argparse.Namespace, dataset: str) -> Path:
    return Path(getattr(args, f"{dataset}_data_root")).expanduser().resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Gallagher-initialized RG-LNS cumulative ablation on "
            "ImageNet-LSEJ-10 erode2, GAP-5, and JPwLEG-5."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASETS),
        default=list(DEFAULT_DATASETS),
        help="Datasets to run, in the requested order",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help=(
            "Subdirectory under eval_result/s1a4_ablation_experiment; "
            "required with --resume or --force"
        ),
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="auto, cpu, cuda, or cuda:N",
    )
    parser.add_argument(
        "--gpu-id",
        default=DEFAULT_GPU_ID,
        type=int,
        help="GPU used when device=auto",
    )
    parser.add_argument(
        "--max-samples",
        default=DEFAULT_MAX_SAMPLES,
        type=int,
        help="Samples per dataset; 0 evaluates the complete test split",
    )
    parser.add_argument(
        "--score-batch-size",
        default=DEFAULT_SCORE_BATCH_SIZE,
        type=int,
    )
    parser.add_argument(
        "--progress-interval",
        default=DEFAULT_PROGRESS_INTERVAL,
        type=int,
    )
    parser.add_argument("--mutual-top-k", default=DEFAULT_MUTUAL_TOP_K, type=int)
    parser.add_argument(
        "--min-component-size",
        default=DEFAULT_MIN_COMPONENT_SIZE,
        type=int,
    )
    parser.add_argument(
        "--component-index",
        default=DEFAULT_COMPONENT_INDEX,
        type=int,
    )
    parser.add_argument(
        "--refinement-rounds",
        default=DEFAULT_REFINEMENT_ROUNDS,
        type=int,
    )
    parser.add_argument(
        "--partial-completion-beam-width",
        default=DEFAULT_PARTIAL_COMPLETION_BEAM_WIDTH,
        type=int,
    )
    parser.add_argument(
        "--save-run-artifacts",
        action="store_true",
        help="Also save evaluator visuals, predictions, and completion-level CSV files",
    )

    parser.add_argument("--lsej-data-root", type=Path, default=DEFAULT_LSEJ_DATA_ROOT)
    parser.add_argument("--gap-data-root", type=Path, default=DEFAULT_GAP_DATA_ROOT)
    parser.add_argument("--jpleg-data-root", type=Path, default=DEFAULT_JPLEG_DATA_ROOT)
    parser.add_argument("--lsej-checkpoint", type=Path, default=DEFAULT_LSEJ_CHECKPOINT)
    parser.add_argument("--gap-checkpoint", type=Path, default=DEFAULT_GAP_CHECKPOINT)
    parser.add_argument("--jpleg-checkpoint", type=Path, default=DEFAULT_JPLEG_CHECKPOINT)

    rerun_group = parser.add_mutually_exclusive_group()
    rerun_group.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed runs with an identical configuration",
    )
    rerun_group.add_argument(
        "--force",
        action="store_true",
        help="Rerun every requested variant inside an existing named run and preserve earlier attempts",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print all commands without creating outputs",
    )
    return parser


def _validate_run_name(run_name: str) -> None:
    if not run_name or run_name in {".", ".."}:
        raise ValueError("run-name must be a non-empty directory name")
    if Path(run_name).name != run_name or "/" in run_name or "\\" in run_name:
        raise ValueError("run-name must not contain a directory separator")


def _resolve_run_root(args: argparse.Namespace) -> Path:
    if args.run_name is not None:
        _validate_run_name(args.run_name)
        return DEFAULT_OUTPUT_ROOT / args.run_name

    base_name = f"table5_{_stamp()}"
    candidate = DEFAULT_OUTPUT_ROOT / base_name
    suffix = 0
    while candidate.exists():
        suffix += 1
        candidate = DEFAULT_OUTPUT_ROOT / f"{base_name}_{suffix}"
    return candidate


def validate_args(args: argparse.Namespace) -> None:
    if (args.resume or args.force) and args.run_name is None:
        raise ValueError("--resume and --force require --run-name")
    if args.gpu_id < 0:
        raise ValueError("gpu-id must be non-negative")
    if args.max_samples < 0:
        raise ValueError("max-samples must be non-negative")
    if args.score_batch_size <= 0:
        raise ValueError("score-batch-size must be positive")
    if args.progress_interval < 0:
        raise ValueError("progress-interval must be non-negative")
    if args.mutual_top_k <= 0:
        raise ValueError("mutual-top-k must be positive")
    if args.min_component_size <= 0:
        raise ValueError("min-component-size must be positive")
    if args.component_index < 0:
        raise ValueError("component-index must be non-negative")
    if args.refinement_rounds <= 0:
        raise ValueError("refinement-rounds must be positive")
    if args.partial_completion_beam_width <= 0:
        raise ValueError("partial-completion-beam-width must be positive")

    # argparse preserves duplicate values supplied after --datasets.  Duplicates
    # would repeat expensive experiments without adding a table column.
    if len(set(args.datasets)) != len(args.datasets):
        raise ValueError("datasets must not contain duplicate entries")

    errors: list[str] = []
    for dataset in args.datasets:
        spec = DATASETS[dataset]
        data_root = _data_root_for(args, dataset)
        checkpoint = _checkpoint_for(args, dataset)
        evaluator_path = MODULE_DIR / f"{spec.module.rsplit('.', 1)[-1]}.py"

        if not data_root.is_dir():
            errors.append(f"{spec.display_name}: data root is missing: {data_root}")
        elif not (data_root / spec.expected_data_entry).exists():
            errors.append(
                f"{spec.display_name}: expected {spec.expected_data_entry!r} "
                f"under data root: {data_root}"
            )
        elif dataset == "lsej" and not (data_root / "lsej_dataloader.py").is_file():
            errors.append(
                f"{spec.display_name}: expected lsej_dataloader.py under data root: "
                f"{data_root}"
            )
        if not checkpoint.is_file():
            errors.append(f"{spec.display_name}: checkpoint is missing: {checkpoint}")
        if not evaluator_path.is_file():
            errors.append(f"{spec.display_name}: evaluator is missing: {evaluator_path}")

    if errors:
        raise FileNotFoundError("Preflight validation failed:\n- " + "\n- ".join(errors))


def _expected_sample_count(args: argparse.Namespace, spec: DatasetSpec) -> int:
    if args.max_samples == 0:
        return spec.full_sample_count
    return min(args.max_samples, spec.full_sample_count)


def _variant_output_dir(run_root: Path, dataset: str, variant: str) -> Path:
    base = run_root / "runs" / dataset / variant
    if not base.exists():
        return base
    candidate = base.with_name(f"{base.name}__{_stamp()}")
    suffix = 0
    while candidate.exists():
        suffix += 1
        candidate = base.with_name(f"{base.name}__{_stamp()}_{suffix}")
    return candidate


def _build_command(
    args: argparse.Namespace,
    spec: DatasetSpec,
    variant: VariantSpec,
    output_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        spec.module,
        spec.selector_argument,
        spec.selector_value,
        "--split",
        "test",
        "--data-root",
        str(_data_root_for(args, spec.key)),
        "--checkpoint",
        str(_checkpoint_for(args, spec.key)),
        "--start-index",
        "0",
        "--max-samples",
        str(args.max_samples),
        "--translation-mode",
        variant.translation_mode,
        "--completion-beam-width",
        str(variant.completion_beam_width),
        "--partial-completion-beam-width",
        str(args.partial_completion_beam_width),
        "--mutual-top-k",
        str(args.mutual_top_k),
        "--component-index",
        str(args.component_index),
        "--min-component-size",
        str(args.min_component_size),
        "--refinement-rounds",
        str(args.refinement_rounds),
        "--score-batch-size",
        str(args.score_batch_size),
        "--postprocess",
        "true",
        "--device",
        args.device,
        "--gpu-id",
        str(args.gpu_id),
        "--output-dir",
        str(output_dir),
        "--save-visuals",
        str(bool(args.save_run_artifacts)).lower(),
        "--visual-limit",
        "10" if args.save_run_artifacts else "0",
        "--save-predictions",
        str(bool(args.save_run_artifacts)).lower(),
        "--save-completion-details",
        str(bool(args.save_run_artifacts)).lower(),
        "--progress-interval",
        str(args.progress_interval),
    ]
    if spec.key == "lsej":
        command.extend(["--loss-type", "hard_triplet"])
    return command


def _run_signature_payload(
    args: argparse.Namespace,
    spec: DatasetSpec,
    variant: VariantSpec,
) -> dict[str, Any]:
    evaluator_path = MODULE_DIR / f"{spec.module.rsplit('.', 1)[-1]}.py"
    result_source_paths = (
        evaluator_path,
        *COMMON_RESULT_SOURCE_PATHS,
        *DATASET_RESULT_SOURCE_PATHS[spec.key],
    )
    if spec.key == "lsej":
        result_source_paths = (
            *result_source_paths,
            _data_root_for(args, spec.key) / "lsej_dataloader.py",
        )
    return {
        "dataset": spec.key,
        "selector": spec.selector_value,
        "data_root": str(_data_root_for(args, spec.key)),
        "checkpoint": _path_fingerprint(_checkpoint_for(args, spec.key)),
        "variant": {
            "translation_mode": variant.translation_mode,
            "completion_beam_width": variant.completion_beam_width,
        },
        "evaluation": {
            "split": "test",
            "start_index": 0,
            "max_samples": args.max_samples,
            "partial_completion_beam_width": args.partial_completion_beam_width,
            "mutual_top_k": args.mutual_top_k,
            "component_index": args.component_index,
            "min_component_size": args.min_component_size,
            "refinement_rounds": args.refinement_rounds,
            "score_batch_size": args.score_batch_size,
            "postprocess": True,
            "device": args.device,
            "gpu_id": args.gpu_id,
            "save_run_artifacts": bool(args.save_run_artifacts),
        },
        "sources": [_source_fingerprint(path) for path in result_source_paths],
    }


def _load_status(path: Path, dataset: str, variant: str) -> dict[str, Any]:
    if path.is_file():
        try:
            payload = _read_json(path)
            if isinstance(payload.get("attempts"), list):
                return payload
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    return {
        "dataset": dataset,
        "variant": variant,
        "state": "not_started",
        "attempts": [],
    }


def _validate_metrics(
    metrics: dict[str, Any],
    *,
    args: argparse.Namespace,
    spec: DatasetSpec,
    variant: VariantSpec,
) -> None:
    observed_selector = metrics.get("task") if spec.key == "lsej" else metrics.get("dataset")
    if observed_selector != spec.selector_value:
        raise ValueError(
            f"Unexpected dataset selector in metrics: {observed_selector!r}; "
            f"expected {spec.selector_value!r}"
        )
    observed_samples = int(metrics.get("evaluated_samples", -1))
    expected_samples = _expected_sample_count(args, spec)
    if observed_samples != expected_samples:
        raise ValueError(
            f"Unexpected sample count for {spec.display_name}: "
            f"{observed_samples}; expected {expected_samples}"
        )

    configuration = metrics.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("metrics.json has no configuration object")
    expected_configuration = {
        "translation_mode": variant.translation_mode,
        "completion_beam_width": variant.completion_beam_width,
        "mutual_top_k": args.mutual_top_k,
        "component_index": args.component_index,
        "min_component_size": args.min_component_size,
        "refinement_rounds": args.refinement_rounds,
    }
    for key, expected in expected_configuration.items():
        if configuration.get(key) != expected:
            raise ValueError(
                f"Unexpected metrics configuration {key}={configuration.get(key)!r}; "
                f"expected {expected!r}"
            )

    for section in (
        "baseline",
        "partial_completion",
        "s1a3_single_path_reference",
        "s1a4",
    ):
        if not isinstance(metrics.get(section), dict) or "PA" not in metrics[section]:
            raise ValueError(f"metrics.json is missing {section}.PA")


def _find_resumable_metrics(
    status: dict[str, Any],
    *,
    signature: str,
    args: argparse.Namespace,
    spec: DatasetSpec,
    variant: VariantSpec,
) -> tuple[dict[str, Any], Path] | None:
    for attempt in reversed(status.get("attempts", [])):
        if attempt.get("state") != "completed" or attempt.get("signature") != signature:
            continue
        metrics_path = Path(str(attempt.get("metrics_path", "")))
        if not metrics_path.is_file():
            continue
        try:
            metrics = _read_json(metrics_path)
            _validate_metrics(
                metrics,
                args=args,
                spec=spec,
                variant=variant,
            )
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        return metrics, metrics_path
    return None


def _preload_resumable_runs(
    args: argparse.Namespace,
    run_root: Path,
) -> dict[tuple[str, str], tuple[dict[str, Any], Path]]:
    """Load every valid completed run before rewriting the top-level summary."""

    if not args.resume:
        return {}
    resumable_runs: dict[tuple[str, str], tuple[dict[str, Any], Path]] = {}
    for dataset in args.datasets:
        spec = DATASETS[dataset]
        for variant in RUN_VARIANTS:
            status_path = run_root / "statuses" / f"{dataset}__{variant.key}.json"
            status = _load_status(status_path, dataset, variant.key)
            signature = _signature(_run_signature_payload(args, spec, variant))
            resumable = _find_resumable_metrics(
                status,
                signature=signature,
                args=args,
                spec=spec,
                variant=variant,
            )
            if resumable is not None:
                resumable_runs[(dataset, variant.key)] = resumable
    return resumable_runs


def _run_subprocess(command: list[str], log_path: Path) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.setdefault("PYTHONUTF8", "1")
    environment.setdefault("PYTHONUNBUFFERED", "1")

    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8", newline="") as log_handle:
        log_handle.write(f"started_at: {_now()}\n")
        log_handle.write(f"cwd: {PROJECT_ROOT}\n")
        log_handle.write(f"command: {subprocess.list2cmdline(command)}\n\n")
        log_handle.flush()

        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log_handle.write(line)
                log_handle.flush()
            return_code = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        finally:
            process.stdout.close()

        elapsed = time.perf_counter() - started
        log_handle.write(f"\nfinished_at: {_now()}\n")
        log_handle.write(f"return_code: {return_code}\n")
        log_handle.write(f"elapsed_seconds: {elapsed:.3f}\n")
    return return_code, elapsed


def _extract_pa(metrics: dict[str, Any], section: str) -> float:
    section_payload = metrics.get(section)
    if not isinstance(section_payload, dict):
        raise ValueError(f"Missing metrics section: {section}")
    pa = float(section_payload["PA"])
    if not 0.0 <= pa <= 1.0:
        raise ValueError(f"Invalid PA value in {section}: {pa}")
    return pa


def _validate_shared_stage_consistency(
    metrics_by_run: dict[tuple[str, str], dict[str, Any]],
) -> None:
    """Ensure the common initial stages are identical across ablation runs."""

    for dataset in DATASETS:
        dataset_metrics = [
            metrics
            for (observed_dataset, _), metrics in metrics_by_run.items()
            if observed_dataset == dataset
        ]
        for section in ("baseline", "partial_completion"):
            values = [_extract_pa(metrics, section) for metrics in dataset_metrics]
            if values and max(values) - min(values) > 1e-12:
                raise ValueError(
                    f"{DATASETS[dataset].display_name} has inconsistent {section}.PA "
                    f"across ablation runs: {values}"
                )


def _aggregate_rows(
    selected_datasets: list[str],
    metrics_by_run: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_spec in TABLE_ROWS:
        pa: dict[str, dict[str, float] | None] = {}
        for dataset in selected_datasets:
            metrics = metrics_by_run.get((dataset, row_spec.run_variant))
            if metrics is None and row_spec.metric_section == "baseline":
                metrics = next(
                    (
                        metrics_by_run[(dataset, variant.key)]
                        for variant in RUN_VARIANTS
                        if (dataset, variant.key) in metrics_by_run
                    ),
                    None,
                )
            if metrics is None:
                pa[dataset] = None
                continue
            fraction = _extract_pa(metrics, row_spec.metric_section)
            pa[dataset] = {
                "fraction": fraction,
                "percent": 100.0 * fraction,
            }
        rows.append({"key": row_spec.key, "label": row_spec.label, "pa": pa})
    return rows


def _write_result_files(
    run_root: Path,
    *,
    args: argparse.Namespace,
    metrics_by_run: dict[tuple[str, str], dict[str, Any]],
    state: str,
) -> None:
    _validate_shared_stage_consistency(metrics_by_run)
    rows = _aggregate_rows(args.datasets, metrics_by_run)
    complete = all(
        row["pa"].get(dataset) is not None
        for row in rows
        for dataset in args.datasets
    )
    results = {
        "experiment": "table5_cumulative_rg_lns_ablation",
        "state": state,
        "complete": complete,
        "updated_at": _now(),
        "output_root": str(run_root),
        "datasets": [
            {
                "key": dataset,
                "display_name": DATASETS[dataset].display_name,
                "data_root": str(_data_root_for(args, dataset)),
                "checkpoint": str(_checkpoint_for(args, dataset)),
                "expected_samples": _expected_sample_count(args, DATASETS[dataset]),
            }
            for dataset in args.datasets
        ],
        "configuration": {
            "initial_solver": "Gallagher",
            "max_samples": args.max_samples,
            "mutual_top_k": args.mutual_top_k,
            "component_index": args.component_index,
            "min_component_size": args.min_component_size,
            "refinement_rounds": args.refinement_rounds,
            "partial_completion_beam_width": args.partial_completion_beam_width,
            "score_batch_size": args.score_batch_size,
            "device": args.device,
            "gpu_id": args.gpu_id,
        },
        "variants": [
            {
                "key": variant.key,
                "label": variant.label,
                "translation_mode": variant.translation_mode,
                "completion_beam_width": variant.completion_beam_width,
            }
            for variant in RUN_VARIANTS
        ],
        "rows": rows,
    }
    _atomic_write_json(run_root / "results.json", results)

    csv_path = run_root / "results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = ["variant", "label", *args.datasets]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row: dict[str, Any] = {"variant": row["key"], "label": row["label"]}
            for dataset in args.datasets:
                value = row["pa"].get(dataset)
                csv_row[dataset] = "" if value is None else f"{value['percent']:.6f}"
            writer.writerow(csv_row)

    display_headers = [DATASETS[dataset].display_name for dataset in args.datasets]
    label_width = max(len("Variant"), *(len(row["label"]) for row in rows))
    column_widths = [max(10, len(header)) for header in display_headers]
    header = "Variant".ljust(label_width)
    for header_text, width in zip(display_headers, column_widths):
        header += " | " + header_text.rjust(width)
    divider = "-" * len(header)
    table_lines = [header, divider]
    for row in rows:
        line = row["label"].ljust(label_width)
        for dataset, width in zip(args.datasets, column_widths):
            value = row["pa"].get(dataset)
            text = "--" if value is None else f"{value['percent']:.1f}"
            line += " | " + text.rjust(width)
        table_lines.append(line)

    summary_lines = [
        "Table 5: cumulative RG-LNS ablation",
        "=" * 80,
        f"state: {state}",
        f"complete: {complete}",
        f"updated_at: {results['updated_at']}",
        f"output_root: {run_root}",
        f"datasets: {', '.join(display_headers)}",
        f"device/gpu_id: {args.device}/{args.gpu_id}",
        f"max_samples: {args.max_samples} (0 means the complete test split)",
        f"configuration: K={args.mutual_top_k}, component={args.component_index}, "
        f"min_size={args.min_component_size}, rounds={args.refinement_rounds}, "
        f"partial_beam={args.partial_completion_beam_width}",
        f"completed evaluator runs: {len(metrics_by_run)}/{len(args.datasets) * len(RUN_VARIANTS)}",
        "metric: PA (%)",
        "",
        *table_lines,
        "",
        "Cumulative variant definitions:",
        "  Reliable components: reciprocal Top-K edges supported by complete 2x2 cycles",
        "  + Reliability-Guided Destroy: fixed component position, greedy completion (B=1)",
        "  + Component Translations: all legal component translations, greedy completion (B=1)",
        "  + Beam-Search Completion: all legal component translations, beam width B=4",
        "  All RG-LNS rows share the same partial-layout completion and strict E1 acceptance rule.",
        "",
        "MGC + Gallagher is intentionally excluded because it is an external compatibility baseline,",
        "not an ablation of the frozen ViT-T compatibility and RG-LNS modules.",
        "",
        "The unrounded PA fractions and percentages are stored in results.json.",
        "Each evaluator command and its stdout/stderr are stored under logs/.",
        "Per-run state and attempt history are stored under statuses/.",
    ]
    _atomic_write_text(run_root / "summary.txt", "\n".join(summary_lines) + "\n")


def _update_experiment_status(
    run_root: Path,
    *,
    args: argparse.Namespace,
    state: str,
    completed_runs: int,
    total_runs: int,
    current_run: str | None = None,
    error: str | None = None,
) -> None:
    payload = {
        "state": state,
        "updated_at": _now(),
        "run_root": str(run_root),
        "datasets": args.datasets,
        "completed_runs": completed_runs,
        "total_runs": total_runs,
        "current_run": current_run,
        "error": error,
    }
    _atomic_write_json(run_root / "experiment_status.json", payload)


def _print_dry_run(args: argparse.Namespace, run_root: Path) -> None:
    print("Preflight validation passed.")
    print(f"Planned output root: {run_root}")
    for dataset in args.datasets:
        spec = DATASETS[dataset]
        for variant in RUN_VARIANTS:
            output_dir = run_root / "runs" / dataset / variant.key
            command = _build_command(args, spec, variant, output_dir)
            print(f"\n[{spec.display_name} / {variant.key}]")
            print(subprocess.list2cmdline(command))


def run(args: argparse.Namespace) -> Path:
    validate_args(args)
    run_root = _resolve_run_root(args)
    if args.dry_run:
        _print_dry_run(args, run_root)
        return run_root

    if (args.resume or args.force) and not run_root.exists():
        action = "resume" if args.resume else "force-rerun"
        raise FileNotFoundError(f"Cannot {action} a missing run directory: {run_root}")
    if run_root.exists() and not (args.resume or args.force):
        raise FileExistsError(
            f"Run directory already exists: {run_root}. "
            "Use --resume to reuse matching completed runs or --force to rerun all variants."
        )
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "logs").mkdir(exist_ok=True)
    (run_root / "statuses").mkdir(exist_ok=True)

    total_runs = len(args.datasets) * len(RUN_VARIANTS)
    resumable_runs = _preload_resumable_runs(args, run_root)
    metrics_by_run: dict[tuple[str, str], dict[str, Any]] = {
        key: metrics for key, (metrics, _) in resumable_runs.items()
    }
    completed_runs = len(metrics_by_run)
    _write_result_files(
        run_root,
        args=args,
        metrics_by_run=metrics_by_run,
        state="running",
    )
    _update_experiment_status(
        run_root,
        args=args,
        state="running",
        completed_runs=completed_runs,
        total_runs=total_runs,
    )

    try:
        for dataset in args.datasets:
            spec = DATASETS[dataset]
            for variant in RUN_VARIANTS:
                run_label = f"{dataset}/{variant.key}"
                status_path = run_root / "statuses" / f"{dataset}__{variant.key}.json"
                status = _load_status(status_path, dataset, variant.key)
                signature_payload = _run_signature_payload(args, spec, variant)
                signature = _signature(signature_payload)

                resumable = resumable_runs.get((dataset, variant.key))
                if resumable is not None:
                    _, metrics_path = resumable
                    print(f"[resume] {run_label}: {metrics_path}")
                    continue

                output_dir = _variant_output_dir(run_root, dataset, variant.key)
                output_dir.parent.mkdir(parents=True, exist_ok=True)
                attempt_stamp = _stamp()
                log_path = run_root / "logs" / f"{dataset}__{variant.key}__{attempt_stamp}.log"
                command = _build_command(args, spec, variant, output_dir)
                attempt: dict[str, Any] = {
                    "attempt": len(status["attempts"]) + 1,
                    "state": "running",
                    "signature": signature,
                    "signature_payload": signature_payload,
                    "started_at": _now(),
                    "command": command,
                    "command_text": subprocess.list2cmdline(command),
                    "log_path": str(log_path),
                    "result_dir": str(output_dir),
                    "metrics_path": str(output_dir / "metrics.json"),
                }
                status["state"] = "running"
                status["attempts"].append(attempt)
                _atomic_write_json(status_path, status)
                _update_experiment_status(
                    run_root,
                    args=args,
                    state="running",
                    completed_runs=completed_runs,
                    total_runs=total_runs,
                    current_run=run_label,
                )

                print("\n" + "=" * 88)
                print(f"Running {spec.display_name}: {variant.label}")
                print(f"Log: {log_path}")
                print("=" * 88)
                started = time.perf_counter()
                try:
                    return_code, elapsed_seconds = _run_subprocess(command, log_path)
                except KeyboardInterrupt:
                    attempt.update(
                        {
                            "state": "interrupted",
                            "finished_at": _now(),
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    )
                    status["state"] = "interrupted"
                    _atomic_write_json(status_path, status)
                    raise
                except Exception as error:
                    attempt.update(
                        {
                            "state": "failed",
                            "finished_at": _now(),
                            "elapsed_seconds": time.perf_counter() - started,
                            "error": str(error),
                        }
                    )
                    status["state"] = "failed"
                    _atomic_write_json(status_path, status)
                    raise

                attempt.update(
                    {
                        "return_code": return_code,
                        "finished_at": _now(),
                        "elapsed_seconds": elapsed_seconds,
                    }
                )
                metrics_path = output_dir / "metrics.json"
                if return_code != 0:
                    attempt["state"] = "failed"
                    attempt["error"] = f"Evaluator exited with code {return_code}"
                    status["state"] = "failed"
                    _atomic_write_json(status_path, status)
                    raise RuntimeError(
                        f"{run_label} failed with exit code {return_code}; see {log_path}"
                    )
                if not metrics_path.is_file():
                    attempt["state"] = "failed"
                    attempt["error"] = f"Missing evaluator output: {metrics_path}"
                    status["state"] = "failed"
                    _atomic_write_json(status_path, status)
                    raise FileNotFoundError(metrics_path)

                metrics = _read_json(metrics_path)
                try:
                    _validate_metrics(
                        metrics,
                        args=args,
                        spec=spec,
                        variant=variant,
                    )
                except ValueError as error:
                    attempt["state"] = "failed"
                    attempt["error"] = str(error)
                    status["state"] = "failed"
                    _atomic_write_json(status_path, status)
                    raise

                attempt["state"] = "completed"
                status["state"] = "completed"
                _atomic_write_json(status_path, status)
                metrics_by_run[(dataset, variant.key)] = metrics
                completed_runs += 1
                _write_result_files(
                    run_root,
                    args=args,
                    metrics_by_run=metrics_by_run,
                    state="running",
                )
                _update_experiment_status(
                    run_root,
                    args=args,
                    state="running",
                    completed_runs=completed_runs,
                    total_runs=total_runs,
                    current_run=None,
                )

    except KeyboardInterrupt:
        _write_result_files(
            run_root,
            args=args,
            metrics_by_run=metrics_by_run,
            state="interrupted",
        )
        _update_experiment_status(
            run_root,
            args=args,
            state="interrupted",
            completed_runs=completed_runs,
            total_runs=total_runs,
            error="Interrupted by user",
        )
        raise
    except Exception as error:
        _write_result_files(
            run_root,
            args=args,
            metrics_by_run=metrics_by_run,
            state="failed",
        )
        _update_experiment_status(
            run_root,
            args=args,
            state="failed",
            completed_runs=completed_runs,
            total_runs=total_runs,
            error=str(error),
        )
        raise

    _write_result_files(
        run_root,
        args=args,
        metrics_by_run=metrics_by_run,
        state="completed",
    )
    _update_experiment_status(
        run_root,
        args=args,
        state="completed",
        completed_runs=completed_runs,
        total_runs=total_runs,
    )
    print(f"\nCompleted Table 5 ablation: {run_root}")
    print((run_root / "summary.txt").read_text(encoding="utf-8"))
    return run_root


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
