"""
Run and summarize the paper runtime experiment for RG-LNS.

Each selected dataset/solver pair is evaluated in a fresh Python process so
that CUDA state and solver caches cannot leak across measurements.  The
dataset evaluators remain the single source of truth: they time compatibility
inference, initial solving, incomplete-layout completion, and RG-LNS itself,
while excluding data loading, ground-truth diagnostics, visualization, and
result serialization.

By default this driver evaluates 200 puzzles (20 warm-up + 180 measured) for
Gallagher, LP, and Pomeranz on ImageNet-LSEJ-10 (2 px), GAP-5, and JPwLEG-5.
Use ``--max-samples 0`` to run the complete test splits.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
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

DEFAULT_DATASETS = ("lsej", "gap", "jpleg")
DEFAULT_SOLVERS = ("gallagher", "lp", "pomeranz")
DEFAULT_MAX_SAMPLES = 200  # 完整测试集改为0
DEFAULT_RUNTIME_WARMUP_SAMPLES = 20
DEFAULT_SCORE_BATCH_SIZE = 128
DEFAULT_PROGRESS_INTERVAL = 25

STAGE_NAMES = (
    "compatibility",
    "initial_solver",
    "partial_completion",
    "rgls",
    "refinement_total",
    "inference_total",
)


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    module_token: str
    selector_argument: str
    selector_value: str
    data_root: Path
    checkpoint: Path


@dataclass(frozen=True)
class SolverSpec:
    key: str
    display_name: str
    module_token: str


DATASETS = {
    "lsej": DatasetSpec(
        key="lsej",
        display_name="ImageNet-LSEJ-10 (2 px)",
        module_token="lsej",
        selector_argument="--task",
        selector_value="grid10_erode2",
        data_root=DEFAULT_LSEJ_DATA_ROOT,
        checkpoint=DEFAULT_LSEJ_CHECKPOINT,
    ),
    "gap": DatasetSpec(
        key="gap",
        display_name="GAP-5",
        module_token="gap",
        selector_argument="--dataset",
        selector_value="GAP-5",
        data_root=DEFAULT_GAP_DATA_ROOT,
        checkpoint=DEFAULT_GAP_CHECKPOINT,
    ),
    "jpleg": DatasetSpec(
        key="jpleg",
        display_name="JPwLEG-5",
        module_token="jpleg",
        selector_argument="--dataset",
        selector_value="jpleg5",
        data_root=DEFAULT_JPLEG_DATA_ROOT,
        checkpoint=DEFAULT_JPLEG_CHECKPOINT,
    ),
}

SOLVERS = {
    "gallagher": SolverSpec("gallagher", "Gallagher", ""),
    "lp": SolverSpec("lp", "LP", "_lp"),
    "pomeranz": SolverSpec("pomeranz", "Pomeranz", "_pomeranz"),
}


def _stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _default_run_root() -> Path:
    return DEFAULT_OUTPUT_ROOT / f"runtime_{_stamp()}"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the RG-LNS runtime experiment and aggregate stage timings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASETS),
        default=list(DEFAULT_DATASETS),
    )
    parser.add_argument(
        "--solvers",
        nargs="+",
        choices=tuple(SOLVERS),
        default=list(DEFAULT_SOLVERS),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help="Puzzles per setting; 0 evaluates the complete test split",
    )
    parser.add_argument(
        "--runtime-warmup-samples",
        type=int,
        default=DEFAULT_RUNTIME_WARMUP_SAMPLES,
        help="Leading puzzles excluded from runtime statistics",
    )
    parser.add_argument("--score-batch-size", type=int, default=DEFAULT_SCORE_BATCH_SIZE)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--progress-interval", type=int, default=DEFAULT_PROGRESS_INTERVAL)
    parser.add_argument("--completion-beam-width", type=int, default=4)
    parser.add_argument("--partial-completion-beam-width", type=int, default=2)
    parser.add_argument("--mutual-top-k", type=int, default=3)
    parser.add_argument("--min-component-size", type=int, default=4)
    parser.add_argument("--component-index", type=int, default=0)
    parser.add_argument("--refinement-rounds", type=int, default=1)
    parser.add_argument(
        "--lsej-data-root", type=Path, default=DEFAULT_LSEJ_DATA_ROOT
    )
    parser.add_argument("--gap-data-root", type=Path, default=DEFAULT_GAP_DATA_ROOT)
    parser.add_argument(
        "--jpleg-data-root", type=Path, default=DEFAULT_JPLEG_DATA_ROOT
    )
    parser.add_argument(
        "--lsej-checkpoint", type=Path, default=DEFAULT_LSEJ_CHECKPOINT
    )
    parser.add_argument("--gap-checkpoint", type=Path, default=DEFAULT_GAP_CHECKPOINT)
    parser.add_argument(
        "--jpleg-checkpoint", type=Path, default=DEFAULT_JPLEG_CHECKPOINT
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--save-run-artifacts",
        action="store_true",
        help="Also save visualizations, predictions, and detailed completion CSVs",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining settings after one subprocess fails",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and write configuration without running evaluations",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_samples < 0 or args.runtime_warmup_samples < 0:
        raise ValueError("max-samples and runtime-warmup-samples must be non-negative")
    for name in (
        "score_batch_size",
        "completion_beam_width",
        "partial_completion_beam_width",
        "mutual_top_k",
        "min_component_size",
        "refinement_rounds",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.component_index < 0:
        raise ValueError("component-index must be non-negative")


def _data_root(args: argparse.Namespace, dataset: str) -> Path:
    return Path(getattr(args, f"{dataset}_data_root")).expanduser().resolve()


def _checkpoint(args: argparse.Namespace, dataset: str) -> Path:
    return Path(getattr(args, f"{dataset}_checkpoint")).expanduser().resolve()


def _preflight(args: argparse.Namespace) -> None:
    errors: list[str] = []
    for dataset in args.datasets:
        root = _data_root(args, dataset)
        checkpoint = _checkpoint(args, dataset)
        if not root.is_dir():
            errors.append(f"{DATASETS[dataset].display_name}: missing data root: {root}")
        if not checkpoint.is_file():
            errors.append(
                f"{DATASETS[dataset].display_name}: missing checkpoint: {checkpoint}"
            )
        for solver in args.solvers:
            module_path = MODULE_DIR / f"{_module_name(dataset, solver).rsplit('.', 1)[-1]}.py"
            if not module_path.is_file():
                errors.append(f"missing evaluator: {module_path}")
    if errors:
        raise FileNotFoundError("Runtime preflight failed:\n- " + "\n- ".join(errors))


def _module_name(dataset: str, solver: str) -> str:
    dataset_token = DATASETS[dataset].module_token
    solver_token = SOLVERS[solver].module_token
    return (
        "s1_based_on_metric_learning."
        f"s1a4_eval_{dataset_token}{solver_token}_profiled_large_neighborhood_reassembly"
    )


def _build_command(
    args: argparse.Namespace,
    *,
    dataset: str,
    solver: str,
    output_dir: Path,
) -> list[str]:
    spec = DATASETS[dataset]
    save_artifacts = str(bool(args.save_run_artifacts)).lower()
    command = [
        sys.executable,
        "-m",
        _module_name(dataset, solver),
        spec.selector_argument,
        spec.selector_value,
        "--split",
        "test",
        "--data-root",
        str(_data_root(args, dataset)),
        "--checkpoint",
        str(_checkpoint(args, dataset)),
        "--start-index",
        "0",
        "--max-samples",
        str(args.max_samples),
        "--completion-beam-width",
        str(args.completion_beam_width),
        "--partial-completion-beam-width",
        str(args.partial_completion_beam_width),
        "--mutual-top-k",
        str(args.mutual_top_k),
        "--translation-mode",
        "all",
        "--component-index",
        str(args.component_index),
        "--min-component-size",
        str(args.min_component_size),
        "--refinement-rounds",
        str(args.refinement_rounds),
        "--score-batch-size",
        str(args.score_batch_size),
        "--runtime-warmup-samples",
        str(args.runtime_warmup_samples),
        "--device",
        args.device,
        "--gpu-id",
        str(args.gpu_id),
        "--output-dir",
        str(output_dir),
        "--save-visuals",
        save_artifacts,
        "--visual-limit",
        "10" if args.save_run_artifacts else "0",
        "--save-predictions",
        save_artifacts,
        "--save-completion-details",
        save_artifacts,
        "--progress-interval",
        str(args.progress_interval),
    ]
    if dataset == "lsej":
        command.extend(["--loss-type", "hard_triplet"])
    return command


def _run_command(command: list[str], log_path: Path) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    child_environment = os.environ.copy()
    child_environment["PYTHONIOENCODING"] = "utf-8"
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(f"command: {subprocess.list2cmdline(command)}\n\n")
        log_handle.flush()
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=child_environment,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_handle.write(line)
            log_handle.flush()
            try:
                print(line, end="")
            except UnicodeEncodeError:
                console_encoding = sys.stdout.encoding or "utf-8"
                safe_line = line.encode(console_encoding, errors="replace").decode(
                    console_encoding,
                    errors="replace",
                )
                print(safe_line, end="")
        return_code = process.wait()
    return return_code, float(time.perf_counter() - started)


def _runtime_rows(
    dataset: str,
    solver: str,
    metrics: dict[str, Any],
    metrics_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    runtime = metrics.get("runtime")
    if not isinstance(runtime, dict) or not isinstance(runtime.get("stages"), dict):
        raise ValueError(f"runtime section missing from {metrics_path}")
    stages = runtime["stages"]
    compact_row: dict[str, Any] = {
        "dataset": DATASETS[dataset].display_name,
        "solver": SOLVERS[solver].display_name,
        "recorded_samples": runtime.get("recorded_samples"),
        "warmup_samples": runtime.get("warmup_samples"),
        "measured_samples": runtime.get("measured_samples"),
    }
    detailed_rows: list[dict[str, Any]] = []
    for stage in STAGE_NAMES:
        statistics = stages.get(stage)
        if not isinstance(statistics, dict):
            raise ValueError(f"runtime stage {stage!r} missing from {metrics_path}")
        compact_row[f"{stage}_mean_seconds"] = statistics.get("mean")
        compact_row[f"{stage}_median_seconds"] = statistics.get("median")
        compact_row[f"{stage}_p95_seconds"] = statistics.get("p95")
        detailed_rows.append(
            {
                "dataset": DATASETS[dataset].display_name,
                "solver": SOLVERS[solver].display_name,
                "stage": stage,
                **statistics,
                "metrics_path": str(metrics_path),
            }
        )
    compact_row["metrics_path"] = str(metrics_path)
    return compact_row, detailed_rows


def _format_summary(
    args: argparse.Namespace,
    run_root: Path,
    rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> str:
    lines = [
        "RG-LNS Runtime Experiment",
        "=" * 80,
        f"output_dir: {run_root}",
        f"datasets: {', '.join(args.datasets)}",
        f"solvers: {', '.join(args.solvers)}",
        f"max_samples: {args.max_samples}",
        f"configured_warmup_samples: {args.runtime_warmup_samples}",
        f"device: {args.device} (gpu_id={args.gpu_id})",
        "Timing excludes data loading, ground-truth diagnostics, visualization, and serialization.",
        "",
        "Mean seconds per puzzle",
        "-----------------------",
    ]
    for row in rows:
        lines.append(
            f"{row['dataset']} | {row['solver']} | n={row['measured_samples']} | "
            f"compat={float(row['compatibility_mean_seconds']):.6f} | "
            f"solver={float(row['initial_solver_mean_seconds']):.6f} | "
            f"partial={float(row['partial_completion_mean_seconds']):.6f} | "
            f"RG-LNS={float(row['rgls_mean_seconds']):.6f} | "
            f"refinement={float(row['refinement_total_mean_seconds']):.6f} | "
            f"total={float(row['inference_total_mean_seconds']):.6f}"
        )
    if failures:
        lines.extend(["", "Failed settings", "---------------"])
        for failure in failures:
            lines.append(
                f"{failure['dataset']} | {failure['solver']} | "
                f"return_code={failure['return_code']} | log={failure['log_path']}"
            )
    lines.extend(
        [
            "",
            "Artifacts",
            "---------",
            "runtime_results.csv: one compact row per dataset/solver setting",
            "runtime_stage_statistics.csv: mean/median/std/p95/total for every stage",
            "results.json: commands, paths, and machine-readable runtime summaries",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> Path:
    _validate_args(args)
    _preflight(args)
    run_root = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else _default_run_root()
    )
    run_root.mkdir(parents=True, exist_ok=True)

    configuration = {
        "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "datasets": list(args.datasets),
        "solvers": list(args.solvers),
        "max_samples": args.max_samples,
        "runtime_warmup_samples": args.runtime_warmup_samples,
        "score_batch_size": args.score_batch_size,
        "device": args.device,
        "gpu_id": args.gpu_id,
        "completion_beam_width": args.completion_beam_width,
        "partial_completion_beam_width": args.partial_completion_beam_width,
        "mutual_top_k": args.mutual_top_k,
        "min_component_size": args.min_component_size,
        "component_index": args.component_index,
        "refinement_rounds": args.refinement_rounds,
        "save_run_artifacts": bool(args.save_run_artifacts),
    }
    _write_json(run_root / "configuration.json", configuration)

    compact_rows: list[dict[str, Any]] = []
    detailed_rows: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for dataset in args.datasets:
        for solver in args.solvers:
            label = f"{DATASETS[dataset].display_name} / {SOLVERS[solver].display_name}"
            output_dir = run_root / "runs" / dataset / solver
            log_path = run_root / "logs" / f"{dataset}__{solver}.log"
            command = _build_command(
                args,
                dataset=dataset,
                solver=solver,
                output_dir=output_dir,
            )
            print(f"\n[{label}]")
            print(subprocess.list2cmdline(command))
            run_record: dict[str, Any] = {
                "dataset": dataset,
                "solver": solver,
                "command": command,
                "output_dir": str(output_dir),
                "log_path": str(log_path),
            }
            if args.dry_run:
                run_record["state"] = "dry_run"
                runs.append(run_record)
                continue

            return_code, wall_seconds = _run_command(command, log_path)
            run_record["return_code"] = return_code
            run_record["subprocess_wall_seconds"] = wall_seconds
            metrics_path = output_dir / "metrics.json"
            if return_code != 0 or not metrics_path.is_file():
                run_record["state"] = "failed"
                failures.append(run_record.copy())
                runs.append(run_record)
                _write_csv(run_root / "runtime_results.csv", compact_rows)
                _write_csv(run_root / "runtime_stage_statistics.csv", detailed_rows)
                _write_json(
                    run_root / "results.json",
                    {"configuration": configuration, "runs": runs},
                )
                (run_root / "summary.txt").write_text(
                    _format_summary(args, run_root, compact_rows, failures),
                    encoding="utf-8",
                )
                if not args.continue_on_error:
                    raise RuntimeError(f"Runtime evaluation failed: {label}; see {log_path}")
                continue

            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            compact_row, stage_rows = _runtime_rows(
                dataset,
                solver,
                metrics,
                metrics_path,
            )
            compact_rows.append(compact_row)
            detailed_rows.extend(stage_rows)
            run_record["state"] = "completed"
            run_record["runtime"] = metrics["runtime"]
            runs.append(run_record)
            _write_csv(run_root / "runtime_results.csv", compact_rows)
            _write_csv(run_root / "runtime_stage_statistics.csv", detailed_rows)
            _write_json(
                run_root / "results.json",
                {"configuration": configuration, "runs": runs},
            )

    _write_csv(run_root / "runtime_results.csv", compact_rows)
    _write_csv(run_root / "runtime_stage_statistics.csv", detailed_rows)
    _write_json(
        run_root / "results.json",
        {"configuration": configuration, "runs": runs},
    )
    summary = _format_summary(args, run_root, compact_rows, failures)
    (run_root / "summary.txt").write_text(summary, encoding="utf-8")
    print(f"\nSaved runtime experiment: {run_root}")
    return run_root


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
