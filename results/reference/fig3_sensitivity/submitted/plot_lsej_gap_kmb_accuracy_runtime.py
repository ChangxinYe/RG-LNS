"""ImageNet-LSEJ 2px 与 GAP-5 / Gallagher 的 PA、RG-LNS 耗时三联图。

直接运行即可，路径以本文件为基准；只依赖 matplotlib。
不覆盖原单数据集脚本及图片，也不修改原始数据和论文。
PA 和时间均直接读取各配置 metrics.json，时间为 runtime.stages.rgls.mean
（秒/张），不包括兼容性计算、初始求解和不完整布局补全。
两组均使用验证集且未剔除预热，但来自不同日期的实验，不能视为同期计时。
颜色区分数据集；实线圆点为 PA（左轴），虚线方点为耗时（右轴）。
耗时量级差异较大，右轴默认采用对数刻度，各面板共用相同范围。
双轴曲线的交点不表示数值相等。TIME_SCALE 可改为 linear 查看线性轴。
"""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext, NullLocator


SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS = (
    {"label": "ImageNet-LSEJ", "benchmark": "lsej10_e2",
     "root": SCRIPT_DIR, "color": "#0072B2", "samples": 1000},
    {"label": "GAP-5", "benchmark": "gap5",
     "root": SCRIPT_DIR.parent / "sensitivity_20260902_221143_375202",
     "color": "#D55E00", "samples": 3000},
)
SOLVER = "gallagher"
SPLIT = "val"
DEFAULTS = {"K": 3, "M": 4, "B": 4}
VALUES = {"K": (1, 2, 3, 4, 5), "M": (4, 6, 8, 10), "B": (1, 2, 4, 8)}
CONFIG_KEYS = {"K": "mutual_top_k", "M": "min_component_size",
               "B": "completion_beam_width"}
OUTPUT_STEM = "lsej_gap_kmb_accuracy_runtime"
FIGURE_SIZE = (4.80, 1.42)  # 宽、高（英寸）；为两行图例保留少量空间。
FONT_SIZE = 9
PNG_DPI = 450
TIME_SCALE = "log"  # log 或 linear；不归一化、不改变原始耗时。
PA_TICK_STEP = 10


def load_series(dataset):
    result = {}
    for parameter, values in VALUES.items():
        points = []
        for value in values:
            config = {**DEFAULTS, parameter: value}
            run = f"k{config['K']}_m{config['M']}_b{config['B']}"
            path = (dataset["root"] / "runs" / dataset["benchmark"]
                    / SOLVER / SPLIT / run / "metrics.json")
            with path.open(encoding="utf-8-sig") as stream:
                metrics = json.load(stream)
            assert metrics["split"] == SPLIT, path
            assert metrics["initial_solver"].lower() == SOLVER, path
            assert metrics["evaluated_samples"] == dataset["samples"], path
            assert metrics["runtime"]["unit"] == "seconds_per_puzzle", path
            assert metrics["runtime"]["warmup_samples"] == 0, path
            assert metrics["runtime"]["measured_samples"] == dataset["samples"], path
            assert all(metrics["configuration"][CONFIG_KEYS[p]] == config[p]
                       for p in DEFAULTS), path
            pa = 100 * float(metrics["s1a4"]["PA"])
            seconds = float(metrics["runtime"]["stages"]["rgls"]["mean"])
            if not (math.isfinite(pa) and 0 <= pa <= 100
                    and math.isfinite(seconds) and seconds > 0):
                raise ValueError(f"Invalid PA or positive runtime: {path}")
            points.append((value, pa, seconds))
        result[parameter] = points
    return result


def main():
    if TIME_SCALE not in ("log", "linear"):
        raise ValueError("TIME_SCALE must be log or linear")
    all_series = [load_series(dataset) for dataset in DATASETS]
    all_points = [point for series in all_series for points in series.values()
                  for point in points]
    pa_low = math.floor(min(p[1] for p in all_points) / PA_TICK_STEP) * PA_TICK_STEP
    pa_high = math.ceil(max(p[1] for p in all_points) / PA_TICK_STEP) * PA_TICK_STEP
    time_low = 10 ** math.floor(math.log10(min(p[2] for p in all_points)))
    time_high = 10 ** math.ceil(math.log10(max(p[2] for p in all_points)))
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": FONT_SIZE,
        "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZE, sharey=True)
    fig.subplots_adjust(left=0.09, right=0.88, bottom=0.24,
                        top=0.72, wspace=0.16)
    legend_handles = []
    for index, (ax, parameter) in enumerate(zip(axes, VALUES)):
        time_ax = ax.twinx()
        for dataset, series in zip(DATASETS, all_series):
            points = series[parameter]
            x, pa, seconds = zip(*points)
            pa_line, = ax.plot(x, pa, color=dataset["color"], linestyle="-",
                              marker="o", linewidth=1.1, markersize=2.8,
                              label=f"{dataset['label']} PA")
            time_line, = time_ax.plot(
                x, seconds, color=dataset["color"], linestyle="--", marker="s",
                markerfacecolor="white", markeredgewidth=0.8,
                linewidth=1.0, markersize=2.8,
                label=f"{dataset['label']} time")
            if index == 0:
                legend_handles.extend([pa_line, time_line])
            print(f"{dataset['label']} {parameter}: " + ", ".join(
                f"{v}: PA={a:.2f}%, RG-LNS={t:.6f}s" for v, a, t in points))
        x = VALUES[parameter]
        margin = (max(x) - min(x)) * 0.10
        ax.set_xlim(min(x) - margin, max(x) + margin)
        ax.set_xticks(x)
        ax.set_xlabel(f"({chr(97 + index)}) ${parameter}$", labelpad=1)
        ax.set_ylim(pa_low, pa_high)
        ax.set_yticks(range(pa_low, pa_high + 1, PA_TICK_STEP))
        time_ax.set_yscale(TIME_SCALE)
        if TIME_SCALE == "log":
            time_ax.set_ylim(time_low, time_high)
            time_ax.yaxis.set_major_locator(LogLocator(base=10))
            time_ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
            time_ax.yaxis.set_minor_locator(NullLocator())
        else:
            time_ax.set_ylim(0, math.ceil(max(p[2] for p in all_points)))
        ax.tick_params(axis="x", length=2, pad=2, width=0.6)
        ax.tick_params(axis="y", length=2, pad=2,
                       left=index == 0, labelleft=index == 0)
        time_ax.tick_params(axis="y", length=2, pad=2,
                            right=index == 2, labelright=index == 2)
        for current in (ax, time_ax):
            current.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(index == 0)
        time_ax.spines["left"].set_visible(False)
        time_ax.spines["right"].set_visible(index == 2)
        time_ax.spines["bottom"].set_visible(False)
        ax.grid(axis="y", color="#e2e2e2", linewidth=0.45)
        ax.set_axisbelow(True)
        if index == 0:
            ax.set_ylabel("PA (%)", labelpad=1)
        if index == 2:
            time_ax.set_ylabel("Time (s/puzzle)", labelpad=2)
    fig.legend(handles=legend_handles, loc="upper center",
               bbox_to_anchor=(0.49, 1.01), ncol=2, frameon=False,
               handlelength=1.6, handletextpad=0.4, columnspacing=1.2,
               labelspacing=0.12, borderaxespad=0)
    for suffix in (".pdf", ".png"):
        path = SCRIPT_DIR / f"{OUTPUT_STEM}{suffix}"
        fig.savefig(path, dpi=PNG_DPI, bbox_inches="tight", pad_inches=0.025)
        print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
