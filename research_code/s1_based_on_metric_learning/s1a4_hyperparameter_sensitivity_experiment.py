"""Run reproducible RG-LNS hyperparameter-sensitivity experiments.

The default experiment performs one-at-a-time (OFAT) sweeps of K, M, and B
on the validation splits of ImageNet-LSEJ-10 (5-pixel erosion) and GAP-5,
both initialized by Gallagher.  The default configuration K=3, M=4, B=4 is
executed once and reused as the default point of all three sensitivity series,
so each benchmark/solver/split setting requires 11 unique evaluator runs.

The command-line interface can expand to all seven paper benchmark settings,
all three initial solvers, validation and test splits, arbitrary value lists,
or the full Cartesian product.  Each evaluator runs in a fresh subprocess.
The driver writes machine-readable plans, per-run status files, logs,
results.json, results.csv, sensitivity.csv (for OFAT), and summary.txt.

Examples
--------
Default 22-run validation experiment::

    python s1a4_hyperparameter_sensitivity_experiment.py

Preview every command without starting an evaluator::

    python s1a4_hyperparameter_sensitivity_experiment.py --dry-run

Run all paper settings and solvers on validation and test splits::

    python s1a4_hyperparameter_sensitivity_experiment.py \
        --benchmarks all --solvers all --splits val test

Run the complete K x M x B Cartesian product::

    python s1a4_hyperparameter_sensitivity_experiment.py \
        --sweep-mode cartesian
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import itertools
import json
import locale
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


SCRIPT_PATH = Path(__file__).resolve()
MODULE_DIR = SCRIPT_PATH.parent
PROJECT_ROOT = MODULE_DIR.parent
DEFAULT_OUTPUT_ROOT = MODULE_DIR / "eval_result" / SCRIPT_PATH.stem

DEFAULT_LSEJ_DATA_ROOT = (
    PROJECT_ROOT / "datasets" / "ImageNet_LSEJ" / "ImageNet_LSEJ"
)
DEFAULT_GAP_DATA_ROOT = PROJECT_ROOT / "datasets" / "GAP_fast"
DEFAULT_JPLEG_DATA_ROOT = PROJECT_ROOT / "datasets" / "MET_Dataset"

# ---------------------------------------------------------------------------
# IDE 一键运行配置
#
# 直接在 IDE 中运行本文件时修改下面这些常量即可；命令行参数或 JSON 配置
# 文件会覆盖相应的默认值。所有序列都必须保留结尾逗号，例如单选应写成
# ("gap5",)，而不是 "gap5" 或 ("gap5")。
# ---------------------------------------------------------------------------

# 要评估的数据设置；可以同时填写任意多个，重复项会自动去除。
# 精确名称：
#   "lsej10_e2"  = ImageNet-LSEJ-10，2-pixel erosion
#   "lsej10_e5"  = ImageNet-LSEJ-10，5-pixel erosion
#   "lsej10_e8"  = ImageNet-LSEJ-10，8-pixel erosion
#   "gap3"       = GAP-3
#   "gap5"       = GAP-5
#   "jpleg3"     = JPwLEG-3
#   "jpleg5"     = JPwLEG-5
# 可用别名（会自动展开）：
#   "lsej10" 或 "lsej" -> lsej10_e2、lsej10_e5、lsej10_e8
#   "gap"                -> gap3、gap5
#   "jpleg"              -> jpleg3、jpleg5
#   "all"                -> 上述全部 7 个精确设置
DEFAULT_BENCHMARKS = ("lsej10_e5", "gap5")

# 初始求解器；可用值："gallagher"、"lp"、"pomeranz"、"all"。
# "all" 会展开为前三种求解器。
DEFAULT_SOLVERS = ("gallagher",)

# 数据划分；可用值："val"、"test"、"all"。
# "all" 会同时运行 val 和 test。超参数选择应以 val 为主，test 用于最终报告。
DEFAULT_SPLITS = ("val",)

# 要扫描的超参数；可用值："k"、"m"、"b"、"all"。
# 可填一个或多个；"all" 等价于 ("k", "m", "b")。
#   K = 构造可靠邻接时使用的 mutual Top-K 排名阈值
#   M = 保留为可靠组件所需的最小碎片数
#   B = repair 阶段的 beam-search width
DEFAULT_SWEEP_PARAMETERS = ("k", "m", "b")

# 扫描方式；仅可填：
#   "one-at-a-time" = 单因素扫描（OFAT）：每次只改变 K、M、B 中的一个；
#                     默认取值列表共产生 11 个不重复配置。
#   "cartesian"     = 对已选择参数取完整笛卡尔积；默认 K/M/B 列表产生
#                     5 x 4 x 4 = 80 个配置。
# 最终 evaluator 运行数还要乘以 数据设置数 x 求解器数 x 数据划分数。
DEFAULT_SWEEP_MODE = "one-at-a-time"

# 各超参数的候选值（必须是正整数）。某参数未包含在
# DEFAULT_SWEEP_PARAMETERS 中时，其候选列表不会参与本次实验。
DEFAULT_K_VALUES = (1, 2, 3, 4, 5)
DEFAULT_M_VALUES = (4, 6, 8, 10)
DEFAULT_B_VALUES = (1, 2, 4, 8)

# 扫描其他参数时保持不变的基准配置。若某参数参与扫描，其默认值必须同时
# 出现在对应的 *_VALUES 中，以便各条敏感性曲线共享同一个基准点。
DEFAULT_K = 3
DEFAULT_M = 4
DEFAULT_B = 4

# 每次 evaluator 最多分析多少个样本：0 = 使用整个划分；正整数 N = 仅用
# 前 N 个样本。小规模调试可先设为 10 或 100，正式实验应设为 0。
DEFAULT_MAX_SAMPLES = 0

# 运行设备；可用值："auto"、"cpu"、"cuda"、"cuda:N"（如 "cuda:1"）。
# "auto" 会在 CUDA 可用时选择 DEFAULT_GPU_ID，否则退回 CPU；"cuda" 同样
# 使用 DEFAULT_GPU_ID；"cuda:N" 已显式指定显卡，因此忽略 DEFAULT_GPU_ID。
DEFAULT_DEVICE = "auto"
DEFAULT_GPU_ID = 0

# 计算 ViT-T 方向嵌入/兼容性分数时的 batch size；必须为正整数。显存不足
# 时减小，显存充足时可增大。它不等于 repair 阶段的 beam width B。
DEFAULT_SCORE_BATCH_SIZE = 128

# 每处理多少个样本打印一次进度：0 = 不打印中间进度；正整数 N = 每 N 个
# 样本打印一次。
DEFAULT_PROGRESS_INTERVAL = 25

# 当初始求解器输出不完整时，自动回填缺失碎片所用的 beam width；必须为
# 正整数。该值只负责“初始解补全”，不是敏感性实验中的 RG-LNS 参数 B。
DEFAULT_PARTIAL_COMPLETION_BEAM_WIDTH = 2

# 若一个初始解含多个可靠组件，选择第几个组件执行 RG-LNS：
# 0 = 最大可靠组件，1 = 第二大可靠组件，依此类推；必须为非负整数。
DEFAULT_COMPONENT_INDEX = 0

# RG-LNS 最多执行的精化轮数；必须为正整数。论文实验默认只执行 1 轮。
DEFAULT_REFINEMENT_ROUNDS = 1

# 仅在启用 --save-run-artifacts 时生效：每次运行最多保存多少个可视化案例；
# 0 = 不限制，正整数 N = 最多保存 N 个。未保存运行产物时该值会被忽略。
DEFAULT_VISUAL_LIMIT = 20


def _checkpoint(*parts: str) -> Path:
    return MODULE_DIR.joinpath("logs", *parts, "checkpoints", "best.pth")


@dataclass(frozen=True)
class BenchmarkSpec:
    key: str
    display_name: str
    family: str
    module_token: str
    selector_argument: str
    selector_value: str
    data_root: Path
    checkpoint: Path
    split_sizes: dict[str, int]
    expected_data_entry: str


@dataclass(frozen=True)
class SolverSpec:
    key: str
    display_name: str
    module_suffix: str


@dataclass(frozen=True, order=True)
class Hyperparameters:
    k: int
    m: int
    b: int

    @property
    def key(self) -> str:
        return f"k{self.k}_m{self.m}_b{self.b}"

    def as_dict(self) -> dict[str, int]:
        return {"k": self.k, "m": self.m, "b": self.b}


@dataclass(frozen=True)
class ExperimentUnit:
    benchmark: str
    solver: str
    split: str
    parameters: Hyperparameters

    @property
    def key(self) -> str:
        return "__".join(
            (self.benchmark, self.solver, self.split, self.parameters.key)
        )


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "lsej10_e2": BenchmarkSpec(
        key="lsej10_e2",
        display_name="ImageNet-LSEJ-10 (2 px)",
        family="lsej",
        module_token="lsej",
        selector_argument="--task",
        selector_value="grid10_erode2",
        data_root=DEFAULT_LSEJ_DATA_ROOT,
        checkpoint=_checkpoint(
            "s1a_train_lsej",
            "grid10_erode2",
            "hard_triplet_d128_s224_k15_2026-07-25-10-22-41",
        ),
        split_sizes={"val": 1000, "test": 2000},
        expected_data_entry="configs",
    ),
    "lsej10_e5": BenchmarkSpec(
        key="lsej10_e5",
        display_name="ImageNet-LSEJ-10 (5 px)",
        family="lsej",
        module_token="lsej",
        selector_argument="--task",
        selector_value="grid10_erode5",
        data_root=DEFAULT_LSEJ_DATA_ROOT,
        checkpoint=_checkpoint(
            "s1a_train_lsej",
            "grid10_erode5",
            "hard_triplet_d128_s224_k15_2026-08-02-21-16-37",
        ),
        split_sizes={"val": 1000, "test": 2000},
        expected_data_entry="configs",
    ),
    "lsej10_e8": BenchmarkSpec(
        key="lsej10_e8",
        display_name="ImageNet-LSEJ-10 (8 px)",
        family="lsej",
        module_token="lsej",
        selector_argument="--task",
        selector_value="grid10_erode8",
        data_root=DEFAULT_LSEJ_DATA_ROOT,
        checkpoint=_checkpoint(
            "s1a_train_lsej",
            "grid10_erode8",
            "hard_triplet_d128_s224_k15_2026-08-02-21-17-06",
        ),
        split_sizes={"val": 1000, "test": 2000},
        expected_data_entry="configs",
    ),
    "gap3": BenchmarkSpec(
        key="gap3",
        display_name="GAP-3",
        family="gap",
        module_token="gap",
        selector_argument="--dataset",
        selector_value="GAP-3",
        data_root=DEFAULT_GAP_DATA_ROOT,
        checkpoint=_checkpoint(
            "s1a_train_gap",
            "GAP-3",
            "hard_triplet_d128_s224_kall_2026-07-31-20-21-04",
        ),
        split_sizes={"val": 3000, "test": 3000},
        expected_data_entry="GAP-3",
    ),
    "gap5": BenchmarkSpec(
        key="gap5",
        display_name="GAP-5",
        family="gap",
        module_token="gap",
        selector_argument="--dataset",
        selector_value="GAP-5",
        data_root=DEFAULT_GAP_DATA_ROOT,
        checkpoint=_checkpoint(
            "s1a_train_gap",
            "GAP-5",
            "hard_triplet_d128_s224_k15_2026-07-31-20-43-26",
        ),
        split_sizes={"val": 3000, "test": 3000},
        expected_data_entry="GAP-5",
    ),
    "jpleg3": BenchmarkSpec(
        key="jpleg3",
        display_name="JPwLEG-3",
        family="jpleg",
        module_token="jpleg",
        selector_argument="--dataset",
        selector_value="jpleg3",
        data_root=DEFAULT_JPLEG_DATA_ROOT,
        checkpoint=_checkpoint(
            "s1a_train_jpleg",
            "jpleg3",
            "hard_triplet_d128_s224_2026-07-27-08-14-53",
        ),
        split_sizes={"val": 1000, "test": 2000},
        expected_data_entry="JPLEG-3",
    ),
    "jpleg5": BenchmarkSpec(
        key="jpleg5",
        display_name="JPwLEG-5",
        family="jpleg",
        module_token="jpleg",
        selector_argument="--dataset",
        selector_value="jpleg5",
        data_root=DEFAULT_JPLEG_DATA_ROOT,
        checkpoint=_checkpoint(
            "s1a_train_jpleg",
            "jpleg5",
            "hard_triplet_d128_s224_2026-07-28-23-27-31",
        ),
        split_sizes={"val": 1000, "test": 2000},
        expected_data_entry="JPLEG-5",
    ),
}

BENCHMARK_ALIASES: dict[str, tuple[str, ...]] = {
    "lsej10": ("lsej10_e2", "lsej10_e5", "lsej10_e8"),
    "lsej": ("lsej10_e2", "lsej10_e5", "lsej10_e8"),
    "gap": ("gap3", "gap5"),
    "jpleg": ("jpleg3", "jpleg5"),
    "all": tuple(BENCHMARKS),
}

SOLVERS: dict[str, SolverSpec] = {
    "gallagher": SolverSpec("gallagher", "Gallagher", ""),
    "lp": SolverSpec("lp", "LP", "_lp"),
    "pomeranz": SolverSpec("pomeranz", "Pomeranz", "_pomeranz"),
}


def _now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _ordered_unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _unique_ints(values: Iterable[int]) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for value in values:
        value = int(value)
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        _atomic_write_text(path, "")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _signature(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _path_fingerprint(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _load_config_defaults(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = _read_json(path.expanduser().resolve())
    result = dict(payload)

    # Accept the documented nested schema as well as direct argparse names.
    default_parameters = payload.get("default_parameters", {})
    if isinstance(default_parameters, dict):
        for key in ("k", "m", "b"):
            if key in default_parameters:
                result[f"default_{key}"] = default_parameters[key]
    values = payload.get("values", {})
    if isinstance(values, dict):
        for key in ("k", "m", "b"):
            if key in values:
                result[f"{key}_values"] = values[key]
    if "sweep_parameters" in payload:
        result["sweep"] = payload["sweep_parameters"]
    return result


def _preparse_config(argv: Sequence[str] | None) -> Path | None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path)
    namespace, _ = parser.parse_known_args(argv)
    return namespace.config


def build_parser(
    config_defaults: dict[str, Any] | None = None,
) -> argparse.ArgumentParser:
    defaults = config_defaults or {}

    def configured(name: str, fallback: Any) -> Any:
        return defaults.get(name, fallback)

    parser = argparse.ArgumentParser(
        description=(
            "Run RG-LNS K/M/B hyperparameter-sensitivity experiments and "
            "aggregate the results."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON configuration; explicit CLI arguments take precedence",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=configured("benchmarks", list(DEFAULT_BENCHMARKS)),
        help=(
            "Exact settings or aliases: lsej10_e2/e5/e8, gap3/5, jpleg3/5, "
            "lsej10, gap, jpleg, all"
        ),
    )
    parser.add_argument(
        "--solvers",
        nargs="+",
        default=configured("solvers", list(DEFAULT_SOLVERS)),
        help="Initial solvers: gallagher, lp, pomeranz, or all",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=configured("splits", list(DEFAULT_SPLITS)),
        help="Evaluation splits: val, test, or all",
    )
    parser.add_argument(
        "--sweep",
        nargs="+",
        default=configured("sweep", list(DEFAULT_SWEEP_PARAMETERS)),
        help="Hyperparameters to sweep: k, m, b, or all",
    )
    parser.add_argument(
        "--sweep-mode",
        choices=("one-at-a-time", "cartesian"),
        default=configured("sweep_mode", DEFAULT_SWEEP_MODE),
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=configured("k_values", list(DEFAULT_K_VALUES)),
    )
    parser.add_argument(
        "--m-values",
        nargs="+",
        type=int,
        default=configured("m_values", list(DEFAULT_M_VALUES)),
    )
    parser.add_argument(
        "--b-values",
        nargs="+",
        type=int,
        default=configured("b_values", list(DEFAULT_B_VALUES)),
    )
    parser.add_argument(
        "--default-k", type=int, default=configured("default_k", DEFAULT_K)
    )
    parser.add_argument(
        "--default-m", type=int, default=configured("default_m", DEFAULT_M)
    )
    parser.add_argument(
        "--default-b", type=int, default=configured("default_b", DEFAULT_B)
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=configured("max_samples", DEFAULT_MAX_SAMPLES),
        help="Samples per evaluator run; 0 means the complete split",
    )
    parser.add_argument(
        "--device", default=configured("device", DEFAULT_DEVICE)
    )
    parser.add_argument(
        "--gpu-id", type=int, default=configured("gpu_id", DEFAULT_GPU_ID)
    )
    parser.add_argument(
        "--score-batch-size",
        type=int,
        default=configured("score_batch_size", DEFAULT_SCORE_BATCH_SIZE),
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=configured("progress_interval", DEFAULT_PROGRESS_INTERVAL),
    )
    parser.add_argument(
        "--partial-completion-beam-width",
        type=int,
        default=configured(
            "partial_completion_beam_width",
            DEFAULT_PARTIAL_COMPLETION_BEAM_WIDTH,
        ),
    )
    parser.add_argument(
        "--component-index",
        type=int,
        default=configured("component_index", DEFAULT_COMPONENT_INDEX),
    )
    parser.add_argument(
        "--refinement-rounds",
        type=int,
        default=configured("refinement_rounds", DEFAULT_REFINEMENT_ROUNDS),
    )
    parser.add_argument(
        "--data-root",
        action="append",
        default=configured("data_root", []),
        metavar="FAMILY=PATH",
        help="Override a family data root; FAMILY is lsej, gap, or jpleg",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=configured("checkpoint", []),
        metavar="BENCHMARK=PATH",
        help="Override one exact benchmark checkpoint, e.g. lsej10_e5=/path/best.pth",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=configured("output_root", DEFAULT_OUTPUT_ROOT),
    )
    parser.add_argument(
        "--run-name",
        default=configured("run_name", None),
        help="Named output directory; required to resume or force an existing run",
    )
    rerun_group = parser.add_mutually_exclusive_group()
    rerun_group.add_argument(
        "--resume",
        "--skip-completed",
        dest="resume",
        action="store_true",
        default=bool(configured("resume", False)),
        help="Reuse completed units with identical signatures",
    )
    rerun_group.add_argument(
        "--force",
        action="store_true",
        default=bool(configured("force", False)),
        help="Rerun requested units inside an existing named run",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=bool(configured("continue_on_error", False)),
    )
    parser.add_argument(
        "--save-run-artifacts",
        action="store_true",
        default=bool(configured("save_run_artifacts", False)),
        help="Also save evaluator visuals, predictions, and completion details",
    )
    parser.add_argument(
        "--visual-limit",
        type=int,
        default=configured("visual_limit", DEFAULT_VISUAL_LIMIT),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=bool(configured("dry_run", False)),
        help="Validate and print the full plan without creating output files",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    config_path = _preparse_config(argv)
    config_defaults = _load_config_defaults(config_path)
    args = build_parser(config_defaults).parse_args(argv)
    args.config = config_path
    return args


def _expand_benchmarks(values: Sequence[str]) -> list[str]:
    expanded: list[str] = []
    for token in values:
        token = token.lower()
        if token in BENCHMARKS:
            expanded.append(token)
        elif token in BENCHMARK_ALIASES:
            expanded.extend(BENCHMARK_ALIASES[token])
        else:
            valid = sorted((*BENCHMARKS, *BENCHMARK_ALIASES))
            raise ValueError(f"Unknown benchmark {token!r}; choices: {valid}")
    return _ordered_unique(expanded)


def _expand_solvers(values: Sequence[str]) -> list[str]:
    expanded: list[str] = []
    for token in values:
        token = token.lower()
        if token == "all":
            expanded.extend(SOLVERS)
        elif token in SOLVERS:
            expanded.append(token)
        else:
            raise ValueError(
                f"Unknown solver {token!r}; choices: {sorted((*SOLVERS, 'all'))}"
            )
    return _ordered_unique(expanded)


def _expand_splits(values: Sequence[str]) -> list[str]:
    expanded: list[str] = []
    for token in values:
        token = token.lower()
        if token == "all":
            expanded.extend(("val", "test"))
        elif token in {"val", "test"}:
            expanded.append(token)
        else:
            raise ValueError("splits must contain only val, test, or all")
    return _ordered_unique(expanded)


def _expand_sweep(values: Sequence[str]) -> list[str]:
    expanded: list[str] = []
    for token in values:
        token = token.lower()
        if token == "all":
            expanded.extend(("k", "m", "b"))
        elif token in {"k", "m", "b"}:
            expanded.append(token)
        else:
            raise ValueError("sweep must contain only k, m, b, or all")
    return _ordered_unique(expanded)


def _parse_assignments(
    values: Sequence[str] | dict[str, Any] | None,
    *,
    valid_keys: set[str],
    label: str,
) -> dict[str, Path]:
    if values is None:
        return {}
    items: list[tuple[str, Any]] = []
    if isinstance(values, dict):
        items.extend(values.items())
    else:
        for raw in values:
            if "=" not in str(raw):
                raise ValueError(f"{label} override must use KEY=PATH: {raw!r}")
            key, path = str(raw).split("=", 1)
            items.append((key, path))
    result: dict[str, Path] = {}
    for raw_key, raw_path in items:
        key = str(raw_key).lower()
        if key not in valid_keys:
            raise ValueError(
                f"Unknown {label} override key {key!r}; choices: {sorted(valid_keys)}"
            )
        if key in result:
            raise ValueError(f"Duplicate {label} override for {key}")
        result[key] = Path(str(raw_path)).expanduser().resolve()
    return result


def _normalize_args(args: argparse.Namespace) -> None:
    args.benchmarks = _expand_benchmarks(args.benchmarks)
    args.solvers = _expand_solvers(args.solvers)
    args.splits = _expand_splits(args.splits)
    args.sweep = _expand_sweep(args.sweep)
    args.k_values = _unique_ints(args.k_values)
    args.m_values = _unique_ints(args.m_values)
    args.b_values = _unique_ints(args.b_values)

    # JSON configuration supports natural dictionaries while argparse uses
    # repeatable KEY=PATH assignments.
    config_payload = _load_config_defaults(args.config) if args.config else {}
    data_root_raw: Sequence[str] | dict[str, Any] | None = args.data_root
    checkpoint_raw: Sequence[str] | dict[str, Any] | None = args.checkpoint
    if not data_root_raw and isinstance(config_payload.get("data_roots"), dict):
        data_root_raw = config_payload["data_roots"]
    if not checkpoint_raw and isinstance(config_payload.get("checkpoints"), dict):
        checkpoint_raw = config_payload["checkpoints"]
    args.data_root_overrides = _parse_assignments(
        data_root_raw,
        valid_keys={"lsej", "gap", "jpleg"},
        label="data-root",
    )
    args.checkpoint_overrides = _parse_assignments(
        checkpoint_raw,
        valid_keys=set(BENCHMARKS),
        label="checkpoint",
    )


def _validate_run_name(run_name: str) -> None:
    if not run_name or run_name in {".", ".."}:
        raise ValueError("run-name must be a non-empty directory name")
    if Path(run_name).name != run_name or "/" in run_name or "\\" in run_name:
        raise ValueError("run-name must not contain directory separators")


def _data_root_for(args: argparse.Namespace, spec: BenchmarkSpec) -> Path:
    return args.data_root_overrides.get(spec.family, spec.data_root).resolve()


def _checkpoint_for(args: argparse.Namespace, spec: BenchmarkSpec) -> Path:
    return args.checkpoint_overrides.get(spec.key, spec.checkpoint).resolve()


def _evaluator_module(spec: BenchmarkSpec, solver: SolverSpec) -> str:
    return (
        "s1_based_on_metric_learning."
        f"s1a4_eval_{spec.module_token}{solver.module_suffix}_"
        "profiled_large_neighborhood_reassembly"
    )


def _evaluator_path(spec: BenchmarkSpec, solver: SolverSpec) -> Path:
    module_name = _evaluator_module(spec, solver).rsplit(".", 1)[-1]
    return MODULE_DIR / f"{module_name}.py"


def _validate_args(args: argparse.Namespace) -> None:
    if (args.resume or args.force) and args.run_name is None:
        raise ValueError("--resume/--skip-completed and --force require --run-name")
    if args.run_name is not None:
        _validate_run_name(args.run_name)
    if args.max_samples < 0:
        raise ValueError("max-samples must be non-negative")
    if args.gpu_id < 0:
        raise ValueError("gpu-id must be non-negative")
    if args.score_batch_size <= 0:
        raise ValueError("score-batch-size must be positive")
    if args.progress_interval < 0:
        raise ValueError("progress-interval must be non-negative")
    if args.partial_completion_beam_width <= 0:
        raise ValueError("partial-completion-beam-width must be positive")
    if args.component_index < 0:
        raise ValueError("component-index must be non-negative")
    if args.refinement_rounds <= 0:
        raise ValueError("refinement-rounds must be positive")
    if args.visual_limit < 0:
        raise ValueError("visual-limit must be non-negative")

    for label, values in (
        ("k-values", args.k_values),
        ("m-values", args.m_values),
        ("b-values", args.b_values),
    ):
        if not values or any(value <= 0 for value in values):
            raise ValueError(f"{label} must contain positive integers")
    for label, value in (
        ("default-k", args.default_k),
        ("default-m", args.default_m),
        ("default-b", args.default_b),
    ):
        if value <= 0:
            raise ValueError(f"{label} must be positive")

    value_lookup = {
        "k": args.k_values,
        "m": args.m_values,
        "b": args.b_values,
    }
    default_lookup = {
        "k": args.default_k,
        "m": args.default_m,
        "b": args.default_b,
    }
    for parameter in args.sweep:
        default_value = default_lookup[parameter]
        if default_value not in value_lookup[parameter]:
            raise ValueError(
                f"default-{parameter}={default_value} must appear in "
                f"{parameter}-values when {parameter} is swept"
            )

    errors: list[str] = []
    checked_evaluators: set[Path] = set()
    for benchmark in args.benchmarks:
        spec = BENCHMARKS[benchmark]
        data_root = _data_root_for(args, spec)
        checkpoint = _checkpoint_for(args, spec)
        if not data_root.is_dir():
            errors.append(f"{spec.display_name}: missing data root: {data_root}")
        elif not (data_root / spec.expected_data_entry).exists():
            errors.append(
                f"{spec.display_name}: expected {spec.expected_data_entry!r} "
                f"under {data_root}"
            )
        if not checkpoint.is_file():
            errors.append(f"{spec.display_name}: missing checkpoint: {checkpoint}")
        for solver_key in args.solvers:
            evaluator_path = _evaluator_path(spec, SOLVERS[solver_key])
            if evaluator_path not in checked_evaluators and not evaluator_path.is_file():
                errors.append(f"Missing evaluator: {evaluator_path}")
            checked_evaluators.add(evaluator_path)
    if errors:
        raise FileNotFoundError("Preflight validation failed:\n- " + "\n- ".join(errors))


def _parameter_plan(args: argparse.Namespace) -> list[Hyperparameters]:
    default = Hyperparameters(args.default_k, args.default_m, args.default_b)
    if args.sweep_mode == "one-at-a-time":
        plan: list[Hyperparameters] = [default]
        for parameter in args.sweep:
            values = getattr(args, f"{parameter}_values")
            for value in values:
                payload = default.as_dict()
                payload[parameter] = value
                plan.append(Hyperparameters(**payload))
    else:
        axes = {
            "k": args.k_values if "k" in args.sweep else [args.default_k],
            "m": args.m_values if "m" in args.sweep else [args.default_m],
            "b": args.b_values if "b" in args.sweep else [args.default_b],
        }
        plan = [
            Hyperparameters(k, m, b)
            for k, m, b in itertools.product(axes["k"], axes["m"], axes["b"])
        ]

    unique: list[Hyperparameters] = []
    seen: set[Hyperparameters] = set()
    for parameters in plan:
        if parameters not in seen:
            seen.add(parameters)
            unique.append(parameters)
    return unique


def _experiment_plan(
    args: argparse.Namespace,
    parameters: Sequence[Hyperparameters],
) -> list[ExperimentUnit]:
    return [
        ExperimentUnit(benchmark, solver, split, configuration)
        for benchmark in args.benchmarks
        for solver in args.solvers
        for split in args.splits
        for configuration in parameters
    ]


def _resolve_run_root(args: argparse.Namespace) -> Path:
    output_root = Path(args.output_root).expanduser().resolve()
    if args.run_name is not None:
        return output_root / args.run_name
    base_name = f"sensitivity_{_stamp()}"
    candidate = output_root / base_name
    suffix = 0
    while candidate.exists():
        suffix += 1
        candidate = output_root / f"{base_name}_{suffix}"
    return candidate


def _expected_sample_count(
    args: argparse.Namespace,
    spec: BenchmarkSpec,
    split: str,
) -> int:
    full_count = spec.split_sizes[split]
    return full_count if args.max_samples == 0 else min(args.max_samples, full_count)


def _build_command(
    args: argparse.Namespace,
    unit: ExperimentUnit,
    output_dir: Path,
) -> list[str]:
    spec = BENCHMARKS[unit.benchmark]
    solver = SOLVERS[unit.solver]
    save = "true" if args.save_run_artifacts else "false"
    return [
        sys.executable,
        "-m",
        _evaluator_module(spec, solver),
        spec.selector_argument,
        spec.selector_value,
        "--split",
        unit.split,
        "--data-root",
        str(_data_root_for(args, spec)),
        "--checkpoint",
        str(_checkpoint_for(args, spec)),
        "--start-index",
        "0",
        "--max-samples",
        str(args.max_samples),
        "--mutual-top-k",
        str(unit.parameters.k),
        "--min-component-size",
        str(unit.parameters.m),
        "--completion-beam-width",
        str(unit.parameters.b),
        "--partial-completion-beam-width",
        str(args.partial_completion_beam_width),
        "--translation-mode",
        "all",
        "--component-index",
        str(args.component_index),
        "--refinement-rounds",
        str(args.refinement_rounds),
        "--score-batch-size",
        str(args.score_batch_size),
        "--device",
        str(args.device),
        "--gpu-id",
        str(args.gpu_id),
        "--output-dir",
        str(output_dir),
        "--save-visuals",
        save,
        "--visual-limit",
        str(args.visual_limit if args.save_run_artifacts else 0),
        "--save-predictions",
        save,
        "--save-completion-details",
        save,
        "--progress-interval",
        str(args.progress_interval),
        "--runtime-warmup-samples",
        "0",
    ]


def _unit_output_dir(run_root: Path, unit: ExperimentUnit) -> Path:
    return (
        run_root
        / "runs"
        / unit.benchmark
        / unit.solver
        / unit.split
        / unit.parameters.key
    )


def _available_output_dir(base: Path) -> Path:
    if not base.exists():
        return base
    candidate = base.with_name(f"{base.name}__{_stamp()}")
    suffix = 0
    while candidate.exists():
        suffix += 1
        candidate = base.with_name(f"{base.name}__{_stamp()}_{suffix}")
    return candidate


def _status_path(run_root: Path, unit: ExperimentUnit) -> Path:
    return run_root / "statuses" / f"{unit.key}.json"


def _load_status(path: Path, unit: ExperimentUnit) -> dict[str, Any]:
    if path.is_file():
        payload = _read_json(path)
        if payload.get("unit_key") != unit.key:
            raise ValueError(f"Status unit mismatch: {path}")
        attempts = payload.get("attempts")
        if not isinstance(attempts, list):
            raise ValueError(f"Invalid attempts in status: {path}")
        return payload
    return {
        "unit_key": unit.key,
        "benchmark": unit.benchmark,
        "solver": unit.solver,
        "split": unit.split,
        "parameters": unit.parameters.as_dict(),
        "state": "pending",
        "attempts": [],
    }


def _run_signature_payload(
    args: argparse.Namespace,
    unit: ExperimentUnit,
) -> dict[str, Any]:
    spec = BENCHMARKS[unit.benchmark]
    solver = SOLVERS[unit.solver]
    evaluator = _evaluator_path(spec, solver)
    return {
        "unit": {
            "benchmark": unit.benchmark,
            "solver": unit.solver,
            "split": unit.split,
            "parameters": unit.parameters.as_dict(),
        },
        "evaluation": {
            "max_samples": args.max_samples,
            "partial_completion_beam_width": args.partial_completion_beam_width,
            "component_index": args.component_index,
            "refinement_rounds": args.refinement_rounds,
            "score_batch_size": args.score_batch_size,
            "translation_mode": "all",
        },
        "data_root": str(_data_root_for(args, spec)),
        "checkpoint": _path_fingerprint(_checkpoint_for(args, spec)),
        "evaluator": _path_fingerprint(evaluator),
    }


def _normalized_observed_split(spec: BenchmarkSpec, split: str) -> str:
    return "valid" if spec.family == "jpleg" and split == "val" else split


def _validate_metrics(
    metrics: dict[str, Any],
    *,
    args: argparse.Namespace,
    unit: ExperimentUnit,
) -> None:
    spec = BENCHMARKS[unit.benchmark]
    selector_key = "task" if spec.family == "lsej" else "dataset"
    if metrics.get(selector_key) != spec.selector_value:
        raise ValueError(
            f"Unexpected {selector_key}={metrics.get(selector_key)!r}; "
            f"expected {spec.selector_value!r}"
        )
    expected_split = _normalized_observed_split(spec, unit.split)
    if metrics.get("split") != expected_split:
        raise ValueError(
            f"Unexpected split={metrics.get('split')!r}; expected {expected_split!r}"
        )
    expected_samples = _expected_sample_count(args, spec, unit.split)
    if int(metrics.get("evaluated_samples", -1)) != expected_samples:
        raise ValueError(
            f"Unexpected evaluated_samples={metrics.get('evaluated_samples')!r}; "
            f"expected {expected_samples}"
        )
    configuration = metrics.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("metrics.json has no configuration object")
    expected_configuration = {
        "completion_beam_width": unit.parameters.b,
        "mutual_top_k": unit.parameters.k,
        "min_component_size": unit.parameters.m,
        "translation_mode": "all",
        "component_index": args.component_index,
        "refinement_rounds": args.refinement_rounds,
        "partial_completion_beam_width": args.partial_completion_beam_width,
    }
    for key, expected in expected_configuration.items():
        if configuration.get(key) != expected:
            raise ValueError(
                f"Unexpected configuration {key}={configuration.get(key)!r}; "
                f"expected {expected!r}"
            )
    final = metrics.get("s1a4")
    if not isinstance(final, dict) or "PA" not in final:
        raise ValueError("metrics.json has no valid s1a4 result")


def _find_resumable_metrics(
    status: dict[str, Any],
    signature: str,
    *,
    args: argparse.Namespace,
    unit: ExperimentUnit,
) -> tuple[dict[str, Any], Path] | None:
    for attempt in reversed(status.get("attempts", [])):
        if attempt.get("state") != "completed":
            continue
        if attempt.get("signature") != signature:
            continue
        metrics_path = Path(str(attempt.get("metrics_path", "")))
        if not metrics_path.is_file():
            continue
        metrics = _read_json(metrics_path)
        try:
            _validate_metrics(metrics, args=args, unit=unit)
        except ValueError:
            continue
        return metrics, metrics_path
    return None


def _run_subprocess(command: list[str], log_path: Path) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    child_encoding = "utf-8" if os.name != "nt" else locale.getpreferredencoding(False)
    child_environment = os.environ.copy()
    child_environment["PYTHONIOENCODING"] = child_encoding
    with log_path.open("w", encoding="utf-8", errors="replace") as log_handle:
        log_handle.write(f"command: {subprocess.list2cmdline(command)}\n\n")
        log_handle.flush()
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding=child_encoding,
            errors="replace",
            bufsize=1,
            env=child_environment,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                console_encoding = sys.stdout.encoding or "utf-8"
                safe_line = line.encode(
                    console_encoding,
                    errors="replace",
                ).decode(console_encoding, errors="replace")
                print(safe_line, end="")
                log_handle.write(line)
                log_handle.flush()
            return_code = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
    return return_code, time.perf_counter() - started


def _metric_value(metrics: dict[str, Any], section: str, key: str) -> float | None:
    payload = metrics.get(section)
    if not isinstance(payload, dict) or payload.get(key) is None:
        return None
    return float(payload[key])


def _result_row(
    unit: ExperimentUnit,
    metrics: dict[str, Any],
    metrics_path: Path,
) -> dict[str, Any]:
    spec = BENCHMARKS[unit.benchmark]
    solver = SOLVERS[unit.solver]
    runtime = metrics.get("runtime") if isinstance(metrics.get("runtime"), dict) else {}
    stage_statistics = (
        runtime.get("stages") if isinstance(runtime.get("stages"), dict) else {}
    )
    refinement = (
        stage_statistics.get("refinement_total")
        if isinstance(stage_statistics.get("refinement_total"), dict)
        else {}
    )
    inference = (
        stage_statistics.get("inference_total")
        if isinstance(stage_statistics.get("inference_total"), dict)
        else {}
    )
    return {
        "unit_key": unit.key,
        "benchmark": unit.benchmark,
        "benchmark_name": spec.display_name,
        "solver": unit.solver,
        "solver_name": solver.display_name,
        "split": unit.split,
        "K": unit.parameters.k,
        "M": unit.parameters.m,
        "B": unit.parameters.b,
        "samples": int(metrics.get("evaluated_samples", 0)),
        "PA": _metric_value(metrics, "s1a4", "PA"),
        "PA_percent": (
            None
            if _metric_value(metrics, "s1a4", "PA") is None
            else 100.0 * float(_metric_value(metrics, "s1a4", "PA"))
        ),
        "AA": _metric_value(metrics, "s1a4", "AA"),
        "AA_percent": (
            None
            if _metric_value(metrics, "s1a4", "AA") is None
            else 100.0 * float(_metric_value(metrics, "s1a4", "AA"))
        ),
        "SRA": _metric_value(metrics, "s1a4", "SRA"),
        "SRA_percent": (
            None
            if _metric_value(metrics, "s1a4", "SRA") is None
            else 100.0 * float(_metric_value(metrics, "s1a4", "SRA"))
        ),
        "baseline_PA": _metric_value(metrics, "baseline", "PA"),
        "baseline_PA_percent": (
            None
            if _metric_value(metrics, "baseline", "PA") is None
            else 100.0 * float(_metric_value(metrics, "baseline", "PA"))
        ),
        "elapsed_seconds": float(metrics.get("elapsed_seconds", 0.0)),
        "refinement_mean_seconds": refinement.get("mean"),
        "inference_mean_seconds": inference.get("mean"),
        "metrics_path": str(metrics_path),
    }


def _ofat_configuration(
    args: argparse.Namespace,
    parameter: str,
    value: int,
) -> Hyperparameters:
    payload = {"k": args.default_k, "m": args.default_m, "b": args.default_b}
    payload[parameter] = value
    return Hyperparameters(**payload)


def _sensitivity_rows(
    args: argparse.Namespace,
    results_by_unit: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if args.sweep_mode != "one-at-a-time":
        return []
    rows: list[dict[str, Any]] = []
    defaults = {"k": args.default_k, "m": args.default_m, "b": args.default_b}
    for benchmark in args.benchmarks:
        for solver in args.solvers:
            for split in args.splits:
                for parameter in args.sweep:
                    for value in getattr(args, f"{parameter}_values"):
                        configuration = _ofat_configuration(args, parameter, value)
                        unit = ExperimentUnit(benchmark, solver, split, configuration)
                        result = results_by_unit.get(unit.key)
                        row: dict[str, Any] = {
                            "benchmark": benchmark,
                            "benchmark_name": BENCHMARKS[benchmark].display_name,
                            "solver": solver,
                            "solver_name": SOLVERS[solver].display_name,
                            "split": split,
                            "parameter": parameter.upper(),
                            "value": value,
                            "is_default": value == defaults[parameter],
                            "K": configuration.k,
                            "M": configuration.m,
                            "B": configuration.b,
                        }
                        for key in (
                            "samples",
                            "PA",
                            "PA_percent",
                            "AA",
                            "AA_percent",
                            "SRA",
                            "SRA_percent",
                            "elapsed_seconds",
                            "refinement_mean_seconds",
                            "inference_mean_seconds",
                            "metrics_path",
                        ):
                            row[key] = None if result is None else result.get(key)
                        rows.append(row)
    return rows


def _experiment_configuration(
    args: argparse.Namespace,
    parameter_plan: Sequence[Hyperparameters],
    units: Sequence[ExperimentUnit],
) -> dict[str, Any]:
    return {
        "experiment": SCRIPT_PATH.stem,
        "created_at": _now(),
        "config_file": None if args.config is None else str(args.config.resolve()),
        "benchmarks": args.benchmarks,
        "solvers": args.solvers,
        "splits": args.splits,
        "sweep_parameters": args.sweep,
        "sweep_mode": args.sweep_mode,
        "default_parameters": {
            "k": args.default_k,
            "m": args.default_m,
            "b": args.default_b,
        },
        "values": {
            "k": args.k_values,
            "m": args.m_values,
            "b": args.b_values,
        },
        "unique_parameter_configurations": [p.as_dict() for p in parameter_plan],
        "unique_parameter_configuration_count": len(parameter_plan),
        "total_evaluator_runs": len(units),
        "max_samples": args.max_samples,
        "device": args.device,
        "gpu_id": args.gpu_id,
        "score_batch_size": args.score_batch_size,
        "progress_interval": args.progress_interval,
        "partial_completion_beam_width": args.partial_completion_beam_width,
        "component_index": args.component_index,
        "refinement_rounds": args.refinement_rounds,
        "save_run_artifacts": args.save_run_artifacts,
        "data_roots": {
            family: str(
                args.data_root_overrides.get(
                    family,
                    {
                        "lsej": DEFAULT_LSEJ_DATA_ROOT,
                        "gap": DEFAULT_GAP_DATA_ROOT,
                        "jpleg": DEFAULT_JPLEG_DATA_ROOT,
                    }[family],
                )
            )
            for family in ("lsej", "gap", "jpleg")
        },
        "checkpoints": {
            benchmark: str(_checkpoint_for(args, BENCHMARKS[benchmark]))
            for benchmark in args.benchmarks
        },
    }


def _write_plan(
    run_root: Path,
    args: argparse.Namespace,
    parameter_plan: Sequence[Hyperparameters],
    units: Sequence[ExperimentUnit],
) -> None:
    configuration = _experiment_configuration(args, parameter_plan, units)
    _atomic_write_json(run_root / "configuration.json", configuration)
    _atomic_write_json(
        run_root / "plan.json",
        {
            "configuration": configuration,
            "units": [
                {
                    "key": unit.key,
                    "benchmark": unit.benchmark,
                    "solver": unit.solver,
                    "split": unit.split,
                    "parameters": unit.parameters.as_dict(),
                }
                for unit in units
            ],
        },
    )


def _format_pa(value: Any) -> str:
    return "--" if value is None else f"{float(value):.2f}"


def _write_result_files(
    run_root: Path,
    *,
    args: argparse.Namespace,
    parameter_plan: Sequence[Hyperparameters],
    units: Sequence[ExperimentUnit],
    results_by_unit: dict[str, dict[str, Any]],
    failed_units: dict[str, str],
    state: str,
) -> None:
    ordered_results = [
        results_by_unit[unit.key] for unit in units if unit.key in results_by_unit
    ]
    sensitivity_rows = _sensitivity_rows(args, results_by_unit)
    complete = len(results_by_unit) == len(units) and not failed_units
    payload = {
        "experiment": SCRIPT_PATH.stem,
        "state": state,
        "complete": complete,
        "updated_at": _now(),
        "output_root": str(run_root),
        "configuration": _experiment_configuration(args, parameter_plan, units),
        "completed_runs": len(results_by_unit),
        "failed_runs": failed_units,
        "results": ordered_results,
        "sensitivity": sensitivity_rows,
    }
    _atomic_write_json(run_root / "results.json", payload)
    _write_csv(run_root / "results.csv", ordered_results)
    _write_csv(run_root / "sensitivity.csv", sensitivity_rows)

    lines = [
        "RG-LNS Hyperparameter Sensitivity Experiment",
        "=" * 80,
        f"state: {state}",
        f"complete: {complete}",
        f"updated_at: {payload['updated_at']}",
        f"output_root: {run_root}",
        f"benchmarks: {', '.join(BENCHMARKS[key].display_name for key in args.benchmarks)}",
        f"solvers: {', '.join(SOLVERS[key].display_name for key in args.solvers)}",
        f"splits: {', '.join(args.splits)}",
        f"sweep_mode: {args.sweep_mode}",
        f"sweep_parameters: {', '.join(parameter.upper() for parameter in args.sweep)}",
        f"default: K={args.default_k}, M={args.default_m}, B={args.default_b}",
        f"K values: {args.k_values}",
        f"M values: {args.m_values}",
        f"B values: {args.b_values}",
        f"unique parameter configurations: {len(parameter_plan)}",
        f"completed evaluator runs: {len(results_by_unit)}/{len(units)}",
        f"failed evaluator runs: {len(failed_units)}",
        f"max_samples: {args.max_samples} (0 means the complete split)",
    ]

    if args.sweep_mode == "one-at-a-time":
        lines.extend(("", "PA sensitivity (%)", "-" * 80))
        for benchmark in args.benchmarks:
            for solver in args.solvers:
                for split in args.splits:
                    lines.append(
                        f"{BENCHMARKS[benchmark].display_name} | "
                        f"{SOLVERS[solver].display_name} | {split}"
                    )
                    for parameter in args.sweep:
                        entries: list[str] = []
                        for value in getattr(args, f"{parameter}_values"):
                            configuration = _ofat_configuration(args, parameter, value)
                            unit = ExperimentUnit(benchmark, solver, split, configuration)
                            result = results_by_unit.get(unit.key)
                            pa = None if result is None else result.get("PA_percent")
                            entries.append(f"{value}:{_format_pa(pa)}")
                        lines.append(f"  {parameter.upper()}: " + " | ".join(entries))
    else:
        lines.extend(
            (
                "",
                "Cartesian results are stored one configuration per row in results.csv.",
            )
        )

    if failed_units:
        lines.extend(("", "Failed units", "-" * 80))
        for unit_key, error in failed_units.items():
            lines.append(f"{unit_key}: {error}")
    lines.extend(
        (
            "",
            "Artifacts:",
            "  results.csv: one row per completed evaluator run",
            "  sensitivity.csv: OFAT plotting table; the default run is reused per series",
            "  results.json: complete machine-readable configuration and results",
            "  plan.json: logical run plan",
            "  logs/: evaluator stdout/stderr",
            "  statuses/: attempts and resume signatures",
        )
    )
    _atomic_write_text(run_root / "summary.txt", "\n".join(lines) + "\n")


def _update_experiment_status(
    run_root: Path,
    *,
    state: str,
    total_runs: int,
    completed_runs: int,
    failed_runs: int,
    current_run: str | None = None,
    error: str | None = None,
) -> None:
    _atomic_write_json(
        run_root / "experiment_status.json",
        {
            "state": state,
            "updated_at": _now(),
            "run_root": str(run_root),
            "total_runs": total_runs,
            "completed_runs": completed_runs,
            "failed_runs": failed_runs,
            "current_run": current_run,
            "error": error,
        },
    )


def _preload_resumable_runs(
    args: argparse.Namespace,
    run_root: Path,
    units: Sequence[ExperimentUnit],
) -> tuple[dict[str, dict[str, Any]], dict[str, Path]]:
    results: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    if not args.resume:
        return results, paths
    for unit in units:
        status_path = _status_path(run_root, unit)
        if not status_path.is_file():
            continue
        status = _load_status(status_path, unit)
        signature = _signature(_run_signature_payload(args, unit))
        resumable = _find_resumable_metrics(
            status,
            signature,
            args=args,
            unit=unit,
        )
        if resumable is None:
            continue
        metrics, metrics_path = resumable
        results[unit.key] = _result_row(unit, metrics, metrics_path)
        paths[unit.key] = metrics_path
    return results, paths


def _print_dry_run(
    args: argparse.Namespace,
    run_root: Path,
    parameter_plan: Sequence[Hyperparameters],
    units: Sequence[ExperimentUnit],
) -> None:
    print("Preflight validation passed.")
    print(f"Planned output root: {run_root}")
    print(f"Unique parameter configurations: {len(parameter_plan)}")
    print(f"Benchmark settings: {len(args.benchmarks)}")
    print(f"Initial solvers: {len(args.solvers)}")
    print(f"Splits: {len(args.splits)}")
    print(f"Total evaluator runs: {len(units)}")
    if len(units) >= 500:
        print("WARNING: this is a large experiment plan.")
    for index, unit in enumerate(units, start=1):
        output_dir = _unit_output_dir(run_root, unit)
        command = _build_command(args, unit, output_dir)
        print(f"\n[{index}/{len(units)}] {unit.key}")
        print(subprocess.list2cmdline(command))


def run(args: argparse.Namespace) -> Path:
    _normalize_args(args)
    _validate_args(args)
    parameter_plan = _parameter_plan(args)
    units = _experiment_plan(args, parameter_plan)
    run_root = _resolve_run_root(args)

    if args.dry_run:
        _print_dry_run(args, run_root, parameter_plan, units)
        return run_root

    if (args.resume or args.force) and not run_root.exists():
        action = "resume" if args.resume else "force-rerun"
        raise FileNotFoundError(f"Cannot {action} missing run directory: {run_root}")
    if run_root.exists() and not (args.resume or args.force):
        raise FileExistsError(
            f"Run directory already exists: {run_root}. Use --resume or --force."
        )

    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "logs").mkdir(exist_ok=True)
    (run_root / "statuses").mkdir(exist_ok=True)
    _write_plan(run_root, args, parameter_plan, units)

    results_by_unit, resumable_paths = _preload_resumable_runs(args, run_root, units)
    failed_units: dict[str, str] = {}
    _write_result_files(
        run_root,
        args=args,
        parameter_plan=parameter_plan,
        units=units,
        results_by_unit=results_by_unit,
        failed_units=failed_units,
        state="running",
    )
    _update_experiment_status(
        run_root,
        state="running",
        total_runs=len(units),
        completed_runs=len(results_by_unit),
        failed_runs=0,
    )

    try:
        for index, unit in enumerate(units, start=1):
            if unit.key in results_by_unit:
                print(f"[resume {index}/{len(units)}] {unit.key}: {resumable_paths[unit.key]}")
                continue

            status_path = _status_path(run_root, unit)
            status = _load_status(status_path, unit)
            signature_payload = _run_signature_payload(args, unit)
            signature = _signature(signature_payload)
            base_output_dir = _unit_output_dir(run_root, unit)
            output_dir = _available_output_dir(base_output_dir)
            attempt_stamp = _stamp()
            log_path = run_root / "logs" / f"{unit.key}__{attempt_stamp}.log"
            command = _build_command(args, unit, output_dir)
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
                state="running",
                total_runs=len(units),
                completed_runs=len(results_by_unit),
                failed_runs=len(failed_units),
                current_run=unit.key,
            )

            print("\n" + "=" * 96)
            print(f"[{index}/{len(units)}] {unit.key}")
            print(f"Log: {log_path}")
            print("=" * 96)
            started = time.perf_counter()
            try:
                return_code, elapsed_seconds = _run_subprocess(command, log_path)
                metrics_path = output_dir / "metrics.json"
                if return_code != 0:
                    raise RuntimeError(f"Evaluator exited with code {return_code}")
                if not metrics_path.is_file():
                    raise FileNotFoundError(f"Missing evaluator output: {metrics_path}")
                metrics = _read_json(metrics_path)
                _validate_metrics(metrics, args=args, unit=unit)
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
                message = str(error)
                attempt.update(
                    {
                        "state": "failed",
                        "finished_at": _now(),
                        "elapsed_seconds": time.perf_counter() - started,
                        "error": message,
                    }
                )
                status["state"] = "failed"
                _atomic_write_json(status_path, status)
                failed_units[unit.key] = message
                _write_result_files(
                    run_root,
                    args=args,
                    parameter_plan=parameter_plan,
                    units=units,
                    results_by_unit=results_by_unit,
                    failed_units=failed_units,
                    state="running_with_errors",
                )
                if not args.continue_on_error:
                    raise RuntimeError(
                        f"{unit.key} failed; see {log_path}: {message}"
                    ) from error
                print(f"[failed; continuing] {unit.key}: {message}")
                continue

            attempt.update(
                {
                    "state": "completed",
                    "finished_at": _now(),
                    "elapsed_seconds": elapsed_seconds,
                }
            )
            status["state"] = "completed"
            _atomic_write_json(status_path, status)
            results_by_unit[unit.key] = _result_row(unit, metrics, metrics_path)
            failed_units.pop(unit.key, None)
            _write_result_files(
                run_root,
                args=args,
                parameter_plan=parameter_plan,
                units=units,
                results_by_unit=results_by_unit,
                failed_units=failed_units,
                state="running" if not failed_units else "running_with_errors",
            )
            _update_experiment_status(
                run_root,
                state="running" if not failed_units else "running_with_errors",
                total_runs=len(units),
                completed_runs=len(results_by_unit),
                failed_runs=len(failed_units),
            )

    except KeyboardInterrupt:
        final_state = "interrupted"
        final_error = "Interrupted by user"
        _write_result_files(
            run_root,
            args=args,
            parameter_plan=parameter_plan,
            units=units,
            results_by_unit=results_by_unit,
            failed_units=failed_units,
            state=final_state,
        )
        _update_experiment_status(
            run_root,
            state=final_state,
            total_runs=len(units),
            completed_runs=len(results_by_unit),
            failed_runs=len(failed_units),
            error=final_error,
        )
        raise
    except Exception as error:
        _write_result_files(
            run_root,
            args=args,
            parameter_plan=parameter_plan,
            units=units,
            results_by_unit=results_by_unit,
            failed_units=failed_units,
            state="failed",
        )
        _update_experiment_status(
            run_root,
            state="failed",
            total_runs=len(units),
            completed_runs=len(results_by_unit),
            failed_runs=len(failed_units),
            error=str(error),
        )
        raise

    final_state = "completed" if not failed_units else "completed_with_errors"
    _write_result_files(
        run_root,
        args=args,
        parameter_plan=parameter_plan,
        units=units,
        results_by_unit=results_by_unit,
        failed_units=failed_units,
        state=final_state,
    )
    _update_experiment_status(
        run_root,
        state=final_state,
        total_runs=len(units),
        completed_runs=len(results_by_unit),
        failed_runs=len(failed_units),
    )
    print(f"\nCompleted hyperparameter experiment: {run_root}")
    print((run_root / "summary.txt").read_text(encoding="utf-8"))
    return run_root


def main(argv: Sequence[str] | None = None) -> None:
    # Keep streamed evaluator output readable in Windows terminals and IDEs.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")
    run(parse_args(argv))


if __name__ == "__main__":
    main()
