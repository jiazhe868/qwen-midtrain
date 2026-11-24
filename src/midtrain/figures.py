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
# The rows contrast what the mixture moves with what it does not (README §1.2): few-shot generation of the trained
# checkpoint, the same tasks after the probe's fine-tune, the knowledge guard, and the math loss as a positive control.
design_rows = [("regression/few_shot_mean/pass@1", {"en": "Δ few-shot pass@1", "zh": "Δ 少样本 pass@1"}),
               ("probe/mean/pass@1", {"en": "Δ probe pass@1", "zh": "Δ 探针 pass@1"}),
               ("regression/knowledge_cloze_mean", {"en": "Δ knowledge", "zh": "Δ 知识"}),
               ("loss/math_web", {"en": "Δ math loss", "zh": "Δ 数学损失"})]


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


# ---------------------------------------------------------------------------------------------------------------
# dose experiment: each run minus its base model, before and after the generic probe's fine-tune
# ---------------------------------------------------------------------------------------------------------------
# the runs the figure shows, top to bottom, with their own colour and marker; runs without results are skipped
dose_run_identities = {
    "dose_reasoning_lr3e-6": {"color": "#2a78d6", "marker": "o", "en": "reasoning mix · 300M · 3e-6",
                              "zh": "推理配比 · 3 亿 · 3e-6"},
    "dose_qaheavy_lr3e-6": {"color": "#1baf7a", "marker": "D", "en": "QA-heavy mix · 300M · 3e-6",
                            "zh": "问答为主 · 3 亿 · 3e-6"},
    "dose_control100m_lr3e-6": {"color": muted_color, "marker": "v", "en": "replay only · 100M · 3e-6",
                                "zh": "只做回放 · 1 亿 · 3e-6"},
    "dose_nosolutions_lr3e-6": {"color": "#6250d6", "marker": "P", "en": "no solutions · 100M · 3e-6",
                                "zh": "去掉解答 · 1 亿 · 3e-6"},
    "dose_distill_lr3e-6": {"color": "#eb6834", "marker": "s", "en": "distillation · 100M · 3e-6",
                            "zh": "蒸馏 · 1 亿 · 3e-6"},
    "dose_distill_lr1e-5": {"color": "#e34948", "marker": "^", "en": "distillation · 100M · 1e-5",
                            "zh": "蒸馏 · 1 亿 · 1e-5"},
    "dose_distill_numina_lr3e-6": {"color": "#e87ba4", "marker": "X", "en": "wider problems · 100M · 3e-6",
                                   "zh": "更广的题目 · 1 亿 · 3e-6"},
}
dose_task_labels = {"gsm8k": "GSM8K", "math500": "MATH-500", "humaneval": "HumanEval", "mbpp": "MBPP",
                    "svamp": "SVAMP", "asdiv_a": "ASDiv-A"}
dose_view_labels = {"few_shot_greedy": {"en": "few-shot", "zh": "少样本"},
                    "generic_probe": {"en": "after fine-tune", "zh": "微调之后"}}


def draw_dose_results(language: str, model_size: str = "0.8B") -> bool:
    """Small multiples: one column per test set, one row per view (the base checkpoint's greedy few-shot accuracy;
    pass@1 after the generic probe). Each mark is one run minus the base model with its 95% interval, from
    results/comparisons/dose_vs_base_<size>.json (`mixing_law.py compare-with-base`)."""
    comparison = read_json(results_directory / "comparisons" / f"dose_vs_base_{model_size}.json")
    if not comparison:
        return False
    shown_runs = [(name, f"{name}_{model_size}") for name in dose_run_identities
                  if f"{name}_{model_size}" in comparison["runs"]]
    if not shown_runs:
        return False
    task_names = [task for task in dose_task_labels
                  if any(task in comparison["runs"][run]["few_shot_greedy"] for _, run in shown_runs)]
    apply_shared_style(language)
    # one x scale per row, so that a gain of 12 points on GSM8K and a change of 2 on HumanEval look as different
    # as they are
    figure, panels = plt.subplots(2, len(task_names), figsize=(1.55 * len(task_names) + 2.2, 0.32 * len(shown_runs) * 2 + 1.4),
                                  sharey=True, sharex="row", squeeze=False)
    rows = {name: row for row, (name, _) in enumerate(reversed(shown_runs))}
    for view_index, view in enumerate(("few_shot_greedy", "generic_probe")):
        for task_index, task_name in enumerate(task_names):
            panel = panels[view_index, task_index]
            has_values = any(comparison["runs"][run_name][view].get(task_name) for _, run_name in shown_runs)
            if not has_values:          # the probe does not test SVAMP and ASDiv-A: leave the cell empty
                panel.set_axis_off()
                continue
            panel.axvline(0, color="#52514e", linewidth=0.8)
            for identity_name, run_name in shown_runs:
                values = comparison["runs"][run_name][view].get(task_name)
                if not values:
                    continue
                difference = values if view == "few_shot_greedy" else values["pass@1"]
                identity = dose_run_identities[identity_name]
                row = rows[identity_name]
                panel.plot([difference["low"], difference["high"]], [row, row], color=identity["color"], linewidth=1.4,
                           solid_capstyle="round")
                panel.plot(difference["difference"], row, marker=identity["marker"], color=identity["color"],
                           markersize=5, markeredgecolor="white", markeredgewidth=0.6, linestyle="none")
            panel.grid(axis="y", visible=False)
            if view_index == 0:
                panel.set_title(dose_task_labels[task_name], fontsize=8.5, color=ink_color)
            if view_index == 1 or not any(comparison["runs"][run_name]["generic_probe"].get(task_name)
                                          for _, run_name in shown_runs):
                panel.set_xlabel("Δ (points)" if language == "en" else "Δ（个点）")
            if task_index == 0:
                panel.set_ylabel(dose_view_labels[view][language])
    panels[0, 0].set_yticks(list(rows.values()), [dose_run_identities[name][language] for name in rows])
    for panel in panels.flat:
        panel.tick_params(axis="y", length=0)
    save_figure(figure, f"dose_results_{model_size}", language)
    return True


# ---------------------------------------------------------------------------------------------------------------
# the paper figures of README §3-5 (11-24): one colour per test set, the same in every figure
# ---------------------------------------------------------------------------------------------------------------
task_identities = {
    "gsm8k": {"color": "#2a78d6", "marker": "o", "label": "GSM8K"},
    "math500": {"color": "#eb6834", "marker": "s", "label": "MATH-500"},
    "svamp": {"color": "#1baf7a", "marker": "D", "label": "SVAMP"},
    "humaneval": {"color": "#6250d6", "marker": "^", "label": "HumanEval"},
    "mbpp": {"color": "#e87ba4", "marker": "v", "label": "MBPP"},
}
model_sizes = ["0.8B", "2B", "4B", "9B"]
words = {
    "few_shot": {"en": "few-shot", "zh": "少样本"},
    "after_fine_tune": {"en": "after fine-tune", "zh": "微调之后"},
    "gain_points": {"en": "Δ vs base (points)", "zh": "相对基座的变化（个点）"},
    "model_size": {"en": "model size", "zh": "模型规模"},
    "peak_learning_rate": {"en": "peak learning rate", "zh": "峰值学习率"},
    "knowledge": {"en": "knowledge (cloze)", "zh": "知识（完形）"},
    "attempts_percent": {"en": "% of attempts", "zh": "尝试的百分比"},
    "right": {"en": "right", "zh": "答对"},
    "cut_off": {"en": "cut off", "zh": "被截断"},
    "first_set": {"en": "GSM8K + MATH (by level)", "zh": "GSM8K + MATH（按难度级别）"},
    "numina_set": {"en": "NuminaMath", "zh": "NuminaMath"},
    "answer_characters": {"en": "answer length (characters)", "zh": "回答长度（字符）"},
    "cut_percent": {"en": "cut at the limit (%)", "zh": "达到长度上限（%）"},
    "loss_change": {"en": "Δ held-out bits per byte (%)", "zh": "留出每字节比特数变化（%）"},
    "pass_at_1": {"en": "Δ pass@1", "zh": "Δ pass@1"},
    "pass_at_16": {"en": "Δ pass@16", "zh": "Δ pass@16"},
}
run_labels = {
    "base": {"en": "base", "zh": "基座"},
    "distill_3e-6": {"en": "distillation, 3e-6", "zh": "蒸馏，3e-6"},
    "distill_1e-5": {"en": "distillation, 1e-5", "zh": "蒸馏，1e-5"},
    "nosolutions": {"en": "without the solutions", "zh": "去掉解答"},
    "control": {"en": "replay only", "zh": "只做回放"},
}


def place_end_labels(panel, entries: list[tuple[float, float, str, str]], minimum_gap_share: float = 0.06) -> None:
    """Write a label at the end of each series; labels closer than `minimum_gap_share` of the y-range are spread
    apart (GSM8K and SVAMP end within 0.2 points of each other at 9B)."""
    if not entries:
        return
    low, high = panel.get_ylim()
    gap = minimum_gap_share * (high - low)
    ordered = sorted(entries, key=lambda entry: entry[1])
    placed = []
    for x, y, label, color in ordered:
        y_label = y if not placed else max(y, placed[-1] + gap)
        placed.append(y_label)
        panel.annotate(label, (x, y), xytext=(x, y_label), textcoords="data", fontsize=7, color=color, va="center",
                       ha="left", annotation_clip=False)


def comparison_of_size(model_size: str) -> dict | None:
    return read_json(results_directory / "comparisons" / f"dose_vs_base_{model_size}.json")


def draw_scaling_gains(language: str) -> bool:
    """The headline: the distillation run at 3e-6 minus its base, at each size, before and after the generic probe."""
    sizes, comparisons = [], {}
    for size in model_sizes:
        comparison = comparison_of_size(size)
        if comparison and f"dose_distill_lr3e-6_{size}" in comparison["runs"]:
            sizes.append(size)
            comparisons[size] = comparison["runs"][f"dose_distill_lr3e-6_{size}"]
    if not sizes:
        return False
    apply_shared_style(language)
    figure, panels = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
    views = [("few_shot", ["gsm8k", "svamp", "math500", "humaneval"]), ("after_fine_tune", ["gsm8k", "math500", "humaneval"])]
    for panel, (view, task_names) in zip(panels, views):
        offsets = np.linspace(-0.18, 0.18, len(task_names))
        label_entries = []
        panel.axhline(0, color="#52514e", linewidth=0.8)
        for offset, task_name in zip(offsets, task_names):
            identity = task_identities[task_name]
            xs, ys, lows, highs = [], [], [], []
            for index, size in enumerate(sizes):
                block = comparisons[size]["few_shot_greedy" if view == "few_shot" else "generic_probe"].get(task_name)
                if not block:
                    continue
                value = block if view == "few_shot" else block["pass@1"]
                xs.append(index + offset)
                ys.append(value["difference"])
                lows.append(value["difference"] - value["low"])
                highs.append(value["high"] - value["difference"])
            if not xs:
                continue
            panel.errorbar(xs, ys, yerr=[lows, highs], color=identity["color"], marker=identity["marker"], markersize=4.5,
                           linewidth=0, elinewidth=1.3, capsize=0, markeredgecolor="white", markeredgewidth=0.5)
            panel.plot(xs, ys, color=identity["color"], linewidth=0.8, alpha=0.5)
            # labels start right of the whole group of points at the last size (offsets reach ±0.18)
            label_entries.append((round(xs[-1]) + 0.3, ys[-1], identity["label"], identity["color"]))
        panel.set_xticks(range(len(sizes)), sizes)
        panel.set_xlim(-0.4, len(sizes) - 1 + 1.1)
        panel.set_title(words[view][language], fontsize=8.5, color=ink_color)
        panel.set_xlabel(words["model_size"][language])
        panel.grid(axis="x", visible=False)
        place_end_labels(panel, label_entries)
    panels[0].set_ylabel(words["gain_points"][language])
    save_figure(figure, "scaling_gains", language)
    return True


def draw_public_text_learning_rate(language: str) -> bool:
    """The first dose sweep: 300M tokens of the reasoning mix at four peak learning rates, minus the base."""
    comparison = comparison_of_size("0.8B")
    if not comparison:
        return False
    rates = [(3e-6, "dose_reasoning_lr3e-6_0.8B"), (1e-5, "dose_reasoning_lr1e-5_0.8B"), (3e-5, "dose_reasoning_lr3e-5_0.8B"),
             (1e-4, "dose_reasoning_lr1e-4_0.8B")]
    rates = [(rate, run) for rate, run in rates if run in comparison["runs"]]
    if len(rates) < 2:
        return False
    apply_shared_style(language)
    figure, panels = plt.subplots(1, 2, figsize=(6.8, 2.8), gridspec_kw={"width_ratios": [2.2, 1]})
    panels[0].axhline(0, color="#52514e", linewidth=0.8)
    for task_name in ("gsm8k", "math500", "humaneval", "mbpp"):
        identity = task_identities[task_name]
        xs = [rate for rate, _ in rates]
        ys = [comparison["runs"][run]["few_shot_greedy"][task_name]["difference"] for _, run in rates]
        panels[0].plot(xs, ys, color=identity["color"], marker=identity["marker"], markersize=4, linewidth=1.5)
        panels[0].annotate(identity["label"], (xs[-1], ys[-1]), xytext=(5, 0), textcoords="offset points", fontsize=7,
                           color=identity["color"], va="center")
    rate_ticks = [rate for rate, _ in rates]
    rate_labels = [f"{rate:.0e}".replace("e-0", "e-") for rate in rate_ticks]
    panels[0].set_xscale("log")
    panels[0].set_xlim(2.2e-6, 2.6e-4)
    panels[0].set_xticks(rate_ticks, rate_labels)
    panels[0].minorticks_off()
    panels[0].set_xlabel(words["peak_learning_rate"][language])
    panels[0].set_ylabel(words["gain_points"][language])
    knowledge = [(rate, comparison["runs"][run].get("knowledge_cloze_change")) for rate, run in rates]
    knowledge = [(rate, value) for rate, value in knowledge if value is not None]
    panels[1].axhline(0, color="#52514e", linewidth=0.8)
    panels[1].plot([rate for rate, _ in knowledge], [value for _, value in knowledge], color=muted_color, marker="o",
                   markersize=4, linewidth=1.5)
    panels[1].set_xscale("log")
    panels[1].set_xticks(rate_ticks, rate_labels)
    panels[1].minorticks_off()
    panels[1].set_xlabel(words["peak_learning_rate"][language])
    panels[1].set_title(words["knowledge"][language], fontsize=8.5, color=ink_color)
    save_figure(figure, "public_text_learning_rate", language)
    return True


def draw_teacher_attempts(language: str) -> bool:
    """How often the teacher got an attempt right, and how often its answer hit the length limit, by group."""
    first = read_json(results_directory / "synthetic_attempts_training_splits.json")
    second = read_json(results_directory / "synthetic_attempts_numina.json")
    if not first:
        return False
    groups = [("GSM8K", first["groups"]["gsm8k"])] + [
        (f"L{level}", first["groups"][f"math/level_{level}"]) for level in range(1, 6)
        if f"math/level_{level}" in first["groups"]]
    if second:
        numina_groups = (("orca", "orca_math"), ("cn_k12", "cn_k12"), ("synth.", "synthetic_math"))
        groups += [(name, second["groups"][key]) for name, key in numina_groups if key in second["groups"]]
    apply_shared_style(language)
    figure, axes = plt.subplots(figsize=(6.6, 2.7))
    positions = [index + (0.6 if index >= 6 else 0) for index in range(len(groups))]
    width = 0.36
    axes.bar([p - width / 2 for p in positions], [100 * g["right_share"] for _, g in groups], width, color="#2a78d6",
             label=words["right"][language])
    axes.bar([p + width / 2 for p in positions], [100 * g["cut_share"] for _, g in groups], width, color="#c9c8c1",
             label=words["cut_off"][language])
    axes.set_xticks(positions, [name for name, _ in groups])
    axes.set_ylabel(words["attempts_percent"][language])
    axes.set_ylim(0, 100)
    axes.grid(axis="x", visible=False)
    axes.legend(frameon=False, fontsize=7, loc="upper right")
    axes.text(2.5, -24, words["first_set"][language], ha="center", fontsize=7, color=muted_color)
    if second:
        axes.text(positions[7], -24, words["numina_set"][language], ha="center", fontsize=7, color=muted_color)
    save_figure(figure, "teacher_attempts", language)
    return True


def draw_answer_length_after_probe(language: str) -> bool:
    """After the generic fine-tune, how long the 0.8B models' answers are and how often they hit the probe's limit."""
    runs = [("base", "base_0.8B", muted_color), ("distill_3e-6", "dose_distill_lr3e-6_0.8B", "#eb6834"),
            ("distill_1e-5", "dose_distill_lr1e-5_0.8B", "#e34948")]
    results = {key: read_json(results_directory / "probe_generic" / f"{run}.json") for key, run, _ in runs}
    if not all(results.values()):
        return False
    apply_shared_style(language)
    figure, panels = plt.subplots(1, 2, figsize=(6.4, 2.6))
    task_names = ["gsm8k", "math500"]
    width = 0.26
    for index, (key, _, color) in enumerate(runs):
        tasks = results[key]["tasks"]
        xs = [position + (index - 1) * width for position in range(len(task_names))]
        panels[0].bar(xs, [tasks[t]["mean_completion_characters"] for t in task_names], width, color=color,
                      label=run_labels[key][language])
        panels[1].bar(xs, [100 * tasks[t]["share_cut_at_token_limit"] for t in task_names], width, color=color)
    for panel, label in zip(panels, ("answer_characters", "cut_percent")):
        panel.set_xticks(range(len(task_names)), [task_identities[t]["label"] for t in task_names])
        panel.set_ylabel(words[label][language])
        panel.grid(axis="x", visible=False)
    panels[0].legend(frameon=False, fontsize=7, loc="upper left")
    save_figure(figure, "answer_length_after_probe", language)
    return True


def draw_loss_by_source(language: str) -> bool:
    """Held-out loss change per source for the distillation run, its ablation and the replay-only control (0.8B)."""
    comparison = comparison_of_size("0.8B")
    runs = [("distill_3e-6", "dose_distill_lr3e-6_0.8B", "#eb6834"), ("nosolutions", "dose_nosolutions_lr3e-6_0.8B", "#6250d6"),
            ("control", "dose_control100m_lr3e-6_0.8B", muted_color)]
    if not comparison or not all(run in comparison["runs"] for _, run, _ in runs):
        return False
    sources = sorted(comparison["runs"][runs[0][1]]["loss_change_percent"],
                     key=lambda source: comparison["runs"][runs[0][1]]["loss_change_percent"][source])
    apply_shared_style(language)
    figure, axes = plt.subplots(figsize=(6.0, 3.4))
    height = 0.26
    for index, (key, run, color) in enumerate(runs):
        changes = comparison["runs"][run]["loss_change_percent"]
        axes.barh([row + (1 - index) * height for row in range(len(sources))], [changes.get(s, 0) for s in sources], height,
                  color=color, label=run_labels[key][language])
    axes.set_yticks(range(len(sources)), sources)
    axes.axvline(0, color="#52514e", linewidth=0.8)
    axes.set_xlabel(words["loss_change"][language])
    axes.grid(axis="y", visible=False)
    axes.legend(frameon=False, fontsize=7, loc="upper left")
    save_figure(figure, "loss_by_source", language)
    return True


def draw_reliability_vs_reach(language: str) -> bool:
    """After the generic probe: the change of pass@1 against the change of pass@16, per task and distillation run."""
    points, markers_used = [], {}
    for size, run, marker in (("0.8B", "dose_distill_lr3e-6_0.8B", "o"), ("0.8B", "dose_distill_lr1e-5_0.8B", "^"),
                              ("2B", "dose_distill_lr3e-6_2B", "s"), ("4B", "dose_distill_lr3e-6_4B", "P"),
                              ("9B", "dose_distill_lr3e-6_9B", "D")):
        comparison = comparison_of_size(size)
        if not comparison or run not in comparison["runs"]:
            continue
        for task_name, block in comparison["runs"][run]["generic_probe"].items():
            if "pass@16" in block:
                points.append((block["pass@1"]["difference"], block["pass@16"]["difference"], task_name, marker))
                markers_used[marker] = f"{size}, {run.split('_lr')[1].split('_')[0]}"
    if not points:
        return False
    apply_shared_style(language)
    figure, axes = plt.subplots(figsize=(4.2, 3.6))
    limit = max(abs(value) for point in points for value in point[:2]) + 1.5
    axes.plot([-limit, limit], [-limit, limit], color="#c9c8c1", linewidth=1, linestyle="--")
    axes.axhline(0, color="#52514e", linewidth=0.6)
    axes.axvline(0, color="#52514e", linewidth=0.6)
    for x, y, task_name, marker in points:
        axes.plot(x, y, marker=marker, color=task_identities[task_name]["color"], markersize=5.5, linestyle="none",
                  markeredgecolor="white", markeredgewidth=0.5)
    for task_name in sorted({point[2] for point in points}):
        axes.plot([], [], marker="o", linestyle="none", color=task_identities[task_name]["color"],
                  label=task_identities[task_name]["label"])
    axes.set_xlim(-limit, limit)
    axes.set_ylim(-limit, limit)
    axes.set_aspect("equal")
    axes.set_xlabel(words["pass_at_1"][language])
    axes.set_ylabel(words["pass_at_16"][language])
    task_legend = axes.legend(frameon=False, fontsize=7, loc="upper left")
    axes.add_artist(task_legend)
    run_handles = [plt.Line2D([], [], marker=marker, linestyle="none", color=muted_color, label=label)
                   for marker, label in markers_used.items()]
    axes.legend(handles=run_handles, frameon=False, fontsize=7, loc="lower right")
    save_figure(figure, "reliability_vs_reach", language)
    return True


def main() -> None:
    drawn = []
    for language in ("en", "zh"):
        for draw in (draw_pilot_loss_change, draw_probe_pass_at_k, draw_design_main_effects, draw_dose_results,
                     draw_scaling_gains, draw_public_text_learning_rate, draw_teacher_attempts,
                     draw_answer_length_after_probe, draw_loss_by_source, draw_reliability_vs_reach):
            if draw(language):
                drawn.append(f"{draw.__name__.removeprefix('draw_')}_{language}")
    print("drew " + ", ".join(drawn) if drawn else "nothing to draw yet")


if __name__ == "__main__":
    main()
