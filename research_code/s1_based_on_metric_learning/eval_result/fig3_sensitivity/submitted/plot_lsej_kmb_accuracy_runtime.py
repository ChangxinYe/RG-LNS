"""绘制 ImageNet-LSEJ 2px 的 K/M/B 双纵轴三联图：左轴 PA，右轴 RG-LNS 耗时。

直接运行本文件即可，不依赖启动时的工作目录；仅需 matplotlib。
读取同目录 sensitivity.csv，另存 PDF/PNG，不覆盖原精度图或修改论文。

计时口径：读取各组 metrics.json 的 runtime.stages.rgls.mean，仅统计 RG-LNS；
不含初始不完整布局补全、兼容性计算和初始求解。PA 仍来自 sensitivity.csv。
这是本次验证实验的内部计时（未剔除预热），不是论文 Runtime 表的协议。
双纵轴表示不同单位，曲线交点没有数值相等的意义。三个面板分别共享
相同的 PA 范围、相同的耗时范围，不对每个面板单独缩放或做归一化。
"""

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------- 可直接在 IDE 中修改的配置 ----------------
SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE = SCRIPT_DIR / "sensitivity.csv"
BENCHMARK = "lsej10_e2"  # ImageNet-LSEJ 2px，Gallagher 初始化，验证集。
SOLVER = "gallagher"
SPLIT = "val"
PARAMETERS = ("K", "M", "B")
# 运行时填入的字段，不读取 CSV 中包含初始布局补全的 refinement_mean_seconds。
TIME_COLUMN = "rglns_mean_seconds"
OUTPUT_STEM = "lsej_kmb_accuracy_runtime"  # 改数据集时可同步改输出名。

# 画布尺寸（宽、高，单位英寸）：加宽并略压低，使横向更舒展、纵向更紧凑。
# 只改变物理间距，不改变数据范围或刻度步长；导出时自动裁去多余白边。
FIGURE_SIZE = (4.40, 1.08)
FONT_SIZE = 9
PNG_DPI = 450
PA_COLOR = "#0072B2"       # 蓝色实线、圆点，对应左轴。
TIME_COLOR = "#D55E00"     # 橙色虚线、方点，对应右轴。
PA_TICK_STEP = 4
TIME_TICK_STEP = 2


def load_series():
    with SOURCE.open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream)
                if row["benchmark"] == BENCHMARK
                and row["solver"] == SOLVER and row["split"] == SPLIT]
    result = {}
    for parameter in PARAMETERS:
        series = sorted((row for row in rows if row["parameter"] == parameter),
                        key=lambda row: int(row["value"]))
        if not series:
            raise ValueError(f"Missing {BENCHMARK}/{SOLVER}/{SPLIT}/{parameter}")
        if len({int(row["value"]) for row in series}) != len(series):
            raise ValueError(f"Duplicate values for {parameter}")
        defaults = [row for row in series if row["is_default"].lower() == "true"]
        if len(defaults) != 1:
            raise ValueError(f"Expected one default point for {parameter}")
        for row in series:
            # CSV 的 metrics_path 是服务器绝对路径；按参数定位本地同目录的运行记录。
            run_name = f"k{row['K']}_m{row['M']}_b{row['B']}"
            metrics_path = (SCRIPT_DIR / "runs" / row["benchmark"] / row["solver"]
                            / row["split"] / run_name / "metrics.json")
            with metrics_path.open(encoding="utf-8-sig") as stream:
                metrics = json.load(stream)
            if metrics["runtime"]["unit"] != "seconds_per_puzzle":
                raise ValueError(f"Unexpected timing unit: {metrics_path}")
            if not math.isclose(float(metrics["s1a4"]["PA"]) * 100,
                                float(row["PA_percent"]), abs_tol=1e-8):
                raise ValueError(f"CSV/metrics PA mismatch: {metrics_path}")
            # 缺少纯精化计时则报错，不能悄悄用全流程或含补全的时间代替。
            row[TIME_COLUMN] = float(metrics["runtime"]["stages"]["rgls"]["mean"])
            pa, seconds = float(row["PA_percent"]), float(row[TIME_COLUMN])
            if not (math.isfinite(pa) and 0 <= pa <= 100
                    and math.isfinite(seconds) and seconds >= 0):
                raise ValueError(f"Invalid PA/time for {parameter}={row['value']}")
            # 验证单因素扫描：另外两个参数必须固定。
            if any(row[p] != defaults[0][p] for p in PARAMETERS if p != parameter):
                raise ValueError(f"Non-swept parameters change in {parameter} series")
        result[parameter] = series
    return result


def main():
    series_by_parameter = load_series()
    all_rows = [row for series in series_by_parameter.values() for row in series]
    pa_values = [float(row["PA_percent"]) for row in all_rows]
    time_values = [float(row[TIME_COLUMN]) for row in all_rows]
    if PA_TICK_STEP <= 0 or TIME_TICK_STEP <= 0:
        raise ValueError("Tick steps must be positive")
    pa_low = math.floor(min(pa_values) / PA_TICK_STEP) * PA_TICK_STEP
    pa_high = math.ceil(max(pa_values) / PA_TICK_STEP) * PA_TICK_STEP
    pa_high = max(pa_high, pa_low + PA_TICK_STEP)
    time_high = max(TIME_TICK_STEP,
                    math.ceil(max(time_values) / TIME_TICK_STEP) * TIME_TICK_STEP)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": FONT_SIZE,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZE, sharey=True)
    # 收紧两侧边距，把宽度实际分配给曲线；底部保留刻度和参数标签空间。
    fig.subplots_adjust(left=0.10, right=0.91, bottom=0.30,
                        top=0.76, wspace=0.16)
    legend_handles = []
    for index, (ax, parameter) in enumerate(zip(axes, PARAMETERS)):
        series = series_by_parameter[parameter]
        x = [int(row["value"]) for row in series]
        pa = [float(row["PA_percent"]) for row in series]
        seconds = [float(row[TIME_COLUMN]) for row in series]
        time_ax = ax.twinx()
        pa_line, = ax.plot(x, pa, color=PA_COLOR, linestyle="-", marker="o",
                           linewidth=1.1, markersize=2.8, label="PA")
        time_line, = time_ax.plot(x, seconds, color=TIME_COLOR, linestyle="--",
                                 marker="s", linewidth=1.1, markersize=2.7,
                                 label="RG-LNS time")
        if index == 0:
            legend_handles = [pa_line, time_line]
        margin = max(max(x) - min(x), 1) * 0.12
        ax.set_xlim(min(x) - margin, max(x) + margin)
        ax.set_xticks(x)
        ax.set_xlabel(f"({chr(97 + index)}) ${parameter}$", labelpad=1)
        ax.set_ylim(pa_low, pa_high)
        ax.set_yticks([pa_low + i * PA_TICK_STEP
                      for i in range(round((pa_high - pa_low) / PA_TICK_STEP) + 1)])
        time_ax.set_ylim(0, time_high)
        time_ax.set_yticks([i * TIME_TICK_STEP
                           for i in range(round(time_high / TIME_TICK_STEP) + 1)])
        ax.tick_params(axis="x", length=2, pad=2, width=0.6)
        ax.tick_params(axis="y", colors=PA_COLOR, length=2, pad=2,
                       left=index == 0, labelleft=index == 0)
        time_ax.tick_params(axis="y", colors=TIME_COLOR, length=2, pad=2,
                            right=index == 2, labelright=index == 2)
        for current in (ax, time_ax):
            current.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(index == 0)
        ax.spines["left"].set_color(PA_COLOR)
        time_ax.spines["left"].set_visible(False)
        time_ax.spines["right"].set_visible(index == 2)
        time_ax.spines["right"].set_color(TIME_COLOR)
        time_ax.spines["bottom"].set_visible(False)
        # 网格只对应左轴，避免两套网格混淆；右轴靠颜色和单位辨识。
        ax.grid(axis="y", color="#e2e2e2", linewidth=0.45)
        ax.set_axisbelow(True)
        if index == 0:
            ax.set_ylabel("PA (%)", color=PA_COLOR, labelpad=1)
        if index == 2:
            time_ax.set_ylabel("Time (s/puzzle)", color=TIME_COLOR, labelpad=2)
        print(f"{parameter}: " + ", ".join(
            f"{value}: PA={accuracy:.1f}%, time={duration:.3f}s"
            for value, accuracy, duration in zip(x, pa, seconds)))

    fig.legend(handles=legend_handles, loc="upper center",
               bbox_to_anchor=(0.50, 0.94), ncol=2, frameon=False,
               handlelength=1.7, handletextpad=0.4, columnspacing=1.1,
               borderaxespad=0)
    for suffix in (".pdf", ".png"):
        destination = SCRIPT_DIR / f"{OUTPUT_STEM}{suffix}"
        fig.savefig(destination, dpi=PNG_DPI, bbox_inches="tight", pad_inches=0.025)
        print(f"Saved: {destination}")
    plt.close(fig)


if __name__ == "__main__":
    main()
