"""Draw the README figures from the result files, in English and Chinese.

Why this file exists. A number typed into plotting code goes stale the moment the run is repeated, and nobody
notices. Every figure here reads its numbers from results/ (training summaries, probe and regression results, the
design analysis), so `python -m midtrain.figures` redraws all of them from the current state of the experiments.
The README text carries the explanation; a figure carries only the shape of the data: no title inside the image,
short axis labels, direct labels at the marks instead of legends where possible, and one fixed colour per run
across every figure. Words that must appear are rendered in both languages (`_en` and `_zh` files).

The colours are the first slots of a categorical palette checked for colour-blind separation; two of them (aqua and
pink) fall below 3:1 contrast on white, which is why every mark carries a direct label or its own marker shape.

    source scripts/env.sh && $MIDTRAIN_PYTHON -m midtrain.figures          # writes assets/*.png (300 dpi) and *.pdf
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (the backend must be chosen before pyplot is imported)
import numpy as np  # noqa: E402

from midtrain.data_sources import repository_root  # noqa: E402

asset_directory = repository_root / "assets"
results_directory = repository_root / "results"
ink_color, muted_color, grid_color = "#0b0b0b", "#898781", "#e1e0d9"
# one colour, marker and pair of labels per run, the same in every figure
run_identities = {
    "base_0.8B": {"color": muted_color, "marker": "o", "en": "base", "zh": "基座"},
    "pilot_replay_only_0.8B_lr1e-5": {"color": "#2a78d6", "marker": "o", "en": "replay, 1e-5", "zh": "回放，1e-5"},
    "pilot_hard_heavy_0.8B_lr1e-5": {"color": "#eb6834", "marker": "s", "en": "math+code, 1e-5", "zh": "数学+代码，1e-5"},
    "pilot_curated_heavy_0.8B_lr1e-5": {"color": "#1baf7a", "marker": "D", "en": "curated, 1e-5", "zh": "精选，1e-5"},
    "pilot_replay_only_0.8B_lr3e-5": {"color": "#e34948", "marker": "v", "en": "replay, 3e-5", "zh": "回放，3e-5"},
    "pilot_replay_only_0.8B_lr3e-6": {"color": "#6250d6", "marker": "^", "en": "replay, 3e-6", "zh": "回放，3e-6"},
    "pilot_hard_heavy_0.8B_lr3e-5": {"color": "#e87ba4", "marker": "P", "en": "math+code, 3e-5", "zh": "数学+代码，3e-5"},
}
source_order = ["replay_web", "replay_web_edu", "wiki_zh", "wiki_en", "textbooks", "papers", "forum_qa", "math_web", "code"]
axis_labels = {
    "loss_change": {"en": "Δ bits per byte (%)", "zh": "每字节比特数变化（%）"},
    "pass_at_k": {"en": "pass@k (%)", "zh": "pass@k（%）"},
}


def apply_shared_style(language: str) -> None:
    plt.rcParams.update({
        "font.family": "Noto Sans CJK SC" if language == "zh" else "DejaVu Sans", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": ink_color,
        "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.grid": True, "grid.color": grid_color,
        "grid.linestyle": "--", "grid.linewidth": 0.6, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "axes.unicode_minus": False})


def save_figure(figure, name: str, language: str) -> None:
    asset_directory.mkdir(exist_ok=True)
    for suffix, settings in ((".png", {"dpi": 300}), (".pdf", {})):
        figure.savefig(asset_directory / f"{name}_{language}{suffix}", bbox_inches="tight", **settings)
    plt.close(figure)


def read_json(file_path: Path) -> dict | None:
    return json.loads(file_path.read_text()) if file_path.exists() else None


# ---------------------------------------------------------------------------------------------------------------
# pilot: how far each run moved the loss on every source
# ---------------------------------------------------------------------------------------------------------------
def draw_pilot_loss_change(language: str) -> bool:
    """One row per source, one marker per pilot run at its change of bits per byte, in percent of the start."""
    runs = {name: read_json(results_directory / "training" / f"{name}.json") for name in run_identities if name.startswith("pilot_")}
    runs = {name: summary for name, summary in runs.items() if summary}
    if not runs:
        return False
    apply_shared_style(language)
    figure, axes = plt.subplots(figsize=(6.4, 3.6))
    row_of_source = {source: row for row, source in enumerate(reversed(source_order))}
    offsets = np.linspace(-0.27, 0.27, len(runs))
    for offset, (run_name, summary) in zip(offsets, runs.items()):
        identity = run_identities[run_name]
        changes = [100 * (summary["validation_after"][source]["bits_per_byte"] / summary["validation_before"][source]["bits_per_byte"] - 1)
                   for source in source_order]
        rows = [row_of_source[source] + offset for source in source_order]
        axes.scatter(changes, rows, s=22, color=identity["color"], marker=identity["marker"], zorder=3, linewidths=0,
                     label=identity[language])
    axes.axvline(0, color="#c3c2b7", linewidth=1, zorder=1)
    axes.set_yticks(range(len(source_order)), list(reversed(source_order)))
    axes.grid(axis="y", visible=False)
    axes.set_xlabel(axis_labels["loss_change"][language])
    axes.legend(frameon=False, loc="upper right", fontsize=8, handletextpad=0.2)       # the top rows end near 2%
    save_figure(figure, "pilot_loss_change", language)
    return True


# ---------------------------------------------------------------------------------------------------------------
# probe: pass@k from 1 to 64 for every probed checkpoint, one panel per task
# ---------------------------------------------------------------------------------------------------------------
def draw_probe_pass_at_k(language: str) -> bool:
    probed = {name: read_json(results_directory / "probe" / f"{name}.json") for name in run_identities}
    probed = {name: result for name, result in probed.items() if result}
    if not probed:
        return False
    task_names = [task for task in ("gsm8k", "math500", "humaneval", "mbpp")
                  if any(task in result["tasks"] for result in probed.values())]
    apply_shared_style(language)
    figure, panels = plt.subplots(1, len(task_names), figsize=(2.0 * len(task_names) + 0.8, 2.6), sharey=True)
    panels = np.atleast_1d(panels)
    for panel, task_name in zip(panels, task_names):
        for run_name, result in probed.items():
            if task_name not in result["tasks"]:
                continue
            task_result = result["tasks"][task_name]
            ks = sorted(int(key.split("@")[1]) for key in task_result if key.startswith("pass@"))
            values = [task_result[f"pass@{k}"]["value"] for k in ks]
            identity = run_identities[run_name]
            panel.plot(ks, values, color=identity["color"], linewidth=1.6, marker=identity["marker"], markersize=3)
            if len(probed) <= 4:        # direct labels only while they do not collide
                panel.annotate(identity[language], (ks[-1], values[-1]), xytext=(3, 0), textcoords="offset points",
                               fontsize=7, color=identity["color"], va="center")
        panel.set_xscale("log", base=2)
        panel.set_xticks([1, 4, 16, 64], ["1", "4", "16", "64"])
        panel.set_ylim(0, 100)
        panel.set_xlabel(f"k · {task_name}")
    panels[0].set_ylabel(axis_labels["pass_at_k"][language])
    if len(probed) > 4:
        handles = [plt.Line2D([], [], color=run_identities[name]["color"], marker=run_identities[name]["marker"],
                              linewidth=1.6, markersize=3, label=run_identities[name][language]) for name in probed]
        figure.legend(handles=handles, frameon=False, loc="upper center", ncol=min(4, len(handles)), fontsize=7,
                      bbox_to_anchor=(0.5, 1.12))
    save_figure(figure, "probe_pass_at_k", language)
    return True


# ---------------------------------------------------------------------------------------------------------------
# design: the mean of each measurement at each level of each factor, for every model size analysed
# ---------------------------------------------------------------------------------------------------------------
size_colors = {"0.8B": "#2a78d6", "2B": "#eb6834"}
factor_labels = {
    "peak_learning_rate": {"en": "peak LR", "zh": "峰值学习率"},
    "replay_share": {"en": "replay share", "zh": "回放占比"},
    "synthetic_share_of_non_replay": {"en": "rewritten share", "zh": "改写占比"},
    "hard_share_of_real": {"en": "hard share", "zh": "难题占比"},
    "math_share_of_hard": {"en": "math share", "zh": "数学占比"},
    "long_context": {"en": "long context", "zh": "长上下文"},
    "schedule": {"en": "schedule", "zh": "时间安排"},
}
level_words = {"none": "无", "random": "随机", "semantic": "语义", "uniform": "均匀", "late": "靠后", "early": "靠前"}
design_rows = [("probe/mean/pass@1", {"en": "Δ pass@1", "zh": "Δ pass@1"}),
               ("probe/mean/pass@16", {"en": "Δ pass@16", "zh": "Δ pass@16"}),
               ("regression/knowledge_cloze_mean", {"en": "Δ knowledge", "zh": "Δ 知识"})]


def level_tick_label(level, language: str) -> str:
    if isinstance(level, float) and 0 < level < 1e-3:
        return f"{level:.0e}".replace("e-0", "e-")            # 3.0e-06 -> 3e-6
    if isinstance(level, str):
        return level_words.get(level, level) if language == "zh" else level
    return f"{level:g}"


def draw_design_main_effects(language: str) -> bool:
    """Small multiples: one row per measurement, one column per factor. Each point is the mean over the runs at one
    level, minus the mean over all runs, so that both model sizes share one axis. The grey band is the range of the
    empty column, the size of an effect of nothing (README §6.2)."""
    import yaml

    design = yaml.safe_load((repository_root / "configs" / "designs" / "screen18.yaml").read_text())
    analyses = {size: read_json(results_directory / "designs" / f"screen18_{size}_analysis.json") for size in size_colors}
    analyses = {size: analysis for size, analysis in analyses.items() if analysis}
    if not analyses:
        return False
    factor_names = list(design["factors"])
    empty_column_names = [name for name in next(iter(analyses.values()))["level_means"][design_rows[0][0]]
                          if name.startswith("empty_column")]
    apply_shared_style(language)
    figure_size = (1.45 * len(factor_names) + 0.6, 1.5 * len(design_rows) + 0.5)
    figure, panels = plt.subplots(len(design_rows), len(factor_names), figsize=figure_size, sharey="row", sharex="col")
    for row_index, (measurement, row_label) in enumerate(design_rows):
        for column_index, factor_name in enumerate(factor_names):
            panel = panels[row_index, column_index]
            levels = design["factors"][factor_name]["levels"]
            for size, analysis in analyses.items():
                level_means = analysis["level_means"].get(measurement)
                if not level_means:
                    continue
                means = np.array(level_means[factor_name]["level_means"])
                grand_mean = float(np.mean([np.mean(level_means[name]["level_means"]) for name in factor_names]))
                empty_ranges = [level_means[name]["range"] for name in empty_column_names if name in level_means]
                if empty_ranges and size == next(iter(analyses)):
                    half_band = max(empty_ranges) / 2
                    panel.axhspan(-half_band, half_band, color=grid_color, alpha=0.7, linewidth=0, zorder=0)
                panel.plot(range(len(means)), means - grand_mean, color=size_colors[size], marker="o", markersize=3.5,
                           linewidth=1.4, zorder=3)
            panel.axhline(0, color="#c3c2b7", linewidth=0.8, zorder=1)
            tick_labels = [level_tick_label(level, language) for level in levels]
            is_wordy = any(len(label) > 5 for label in tick_labels)       # "semantic" next to "random" would collide
            panel.set_xticks(range(len(levels)), tick_labels, fontsize=6.5 if is_wordy else 7, rotation=25 if is_wordy else 0)
            panel.set_xlim(-0.4, len(levels) - 0.6)
            panel.grid(axis="x", visible=False)
            if row_index == len(design_rows) - 1:
                panel.set_xlabel(factor_labels[factor_name][language], fontsize=8)
        panels[row_index, 0].set_ylabel(row_label[language])
    if len(analyses) > 1:        # one legend for the two sizes; with one size the caption names it
        handles = [plt.Line2D([], [], color=size_colors[size], marker="o", markersize=3.5, linewidth=1.4, label=size)
                   for size in analyses]
        figure.legend(handles=handles, frameon=False, loc="upper center", ncol=len(handles), fontsize=8, bbox_to_anchor=(0.5, 1.03))
    figure.tight_layout()
    save_figure(figure, "design_main_effects", language)
    return True


def main() -> None:
    drawn = []
    for language in ("en", "zh"):
        for draw in (draw_pilot_loss_change, draw_probe_pass_at_k, draw_design_main_effects):
            if draw(language):
                drawn.append(f"{draw.__name__.removeprefix('draw_')}_{language}")
    print("drew " + ", ".join(drawn) if drawn else "nothing to draw yet")


if __name__ == "__main__":
    main()
