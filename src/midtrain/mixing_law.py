"""Design the mixture experiments, queue them, and turn their results into a mixing law.

Why this file exists. The mixture has many knobs (how much replay, how much rewritten text, how much math and code,
long sequences or not, when to spend the non-replay share, the learning rate), the shares sum to one, and the knobs
interact. Sweeping one knob at a time needs many runs and cannot see an interaction. An orthogonal array varies all
knobs at once: every level of every factor meets every level of every other factor equally often, so the effect of
one factor is an average over all settings of the others. The L18 array used here holds one factor with two levels
and up to seven with three in 18 runs; the full grid would need 2 x 3^6 = 1,458.

What the file does, in the order the stages run:
    design    a design file (configs/designs/*.yaml) -> 18 recipes with exact shares, and the table of runs
    queue     one task per run for a model size: continued pretraining, then the regression suite, then the probe
    analyze   level means per factor; a regression on the absolute token shares (the mixing law); the same law for
              every measurement side by side (the interference matrix); the best mixture inside the tested ranges
              under the constraint that the regression suite does not fall; rank agreement between two model sizes
    compare   two runs question by question: both answered the same questions with the same sampling, so their
              difference is known far better than either score alone
    compare-with-base  every run of one size minus its base model, on the greedy final evaluation, the generic
              probe, the knowledge guard and the losses (the dose experiment, README §1.2)

The factors are ratios (replay share of all tokens, rewritten share of the non-replay text, ...) because the
questions are about ratios. The law is fitted on absolute shares, with replay as what remains, so a coefficient
reads: the change of a measurement when this component takes one unit of share from replay (README §6.3).

Commands (after `source scripts/env.sh`):
    $MIDTRAIN_PYTHON -m midtrain.mixing_law design configs/designs/screen18.yaml
    $MIDTRAIN_PYTHON -m midtrain.mixing_law queue configs/designs/screen18.yaml --model-size 0.8B \
        --queue $MIDTRAIN_DATA/queue_local
    $MIDTRAIN_PYTHON -m midtrain.mixing_law analyze configs/designs/screen18.yaml --model-size 0.8B
    $MIDTRAIN_PYTHON -m midtrain.mixing_law analyze configs/designs/screen18.yaml --model-size 0.8B --compare-with 2B
    $MIDTRAIN_PYTHON -m midtrain.mixing_law compare pilot_hard_heavy_0.8B_lr1e-5 pilot_replay_only_0.8B_lr1e-5
    $MIDTRAIN_PYTHON -m midtrain.mixing_law compare-with-base --model-size 0.8B
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import yaml

from midtrain.data_sources import data_root, repository_root

# The L18 orthogonal array (Taguchi), levels counted from 0: column 0 has two levels, columns 1 to 7 have three.
# check_array_is_orthogonal() runs at import, so a typing mistake in this table cannot go unnoticed.
l18_array = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1, 1, 1], [0, 0, 2, 2, 2, 2, 2, 2],
    [0, 1, 0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 2, 2, 0, 0], [0, 1, 2, 2, 0, 0, 1, 1],
    [0, 2, 0, 1, 0, 2, 1, 2], [0, 2, 1, 2, 1, 0, 2, 0], [0, 2, 2, 0, 2, 1, 0, 1],
    [1, 0, 0, 2, 2, 1, 1, 0], [1, 0, 1, 0, 0, 2, 2, 1], [1, 0, 2, 1, 1, 0, 0, 2],
    [1, 1, 0, 1, 2, 0, 2, 1], [1, 1, 1, 2, 0, 1, 0, 2], [1, 1, 2, 0, 1, 2, 1, 0],
    [1, 2, 0, 2, 1, 2, 0, 1], [1, 2, 1, 0, 2, 0, 1, 2], [1, 2, 2, 1, 0, 1, 2, 0]])

# the components of the law, in the order of its columns; replay is what remains and has no column
law_component_names = ["synthetic", "curated", "math", "code", "long_random", "long_semantic"]
# per model size: the base checkpoint folder under $MIDTRAIN_MODELS, where the 32-bit master weights live (they fit
# on the GPU next to the 0.8B model only: 12 GB with the Adam moments), and how many GPUs one run takes
settings_of_model_size = {
    "0.8B": {"folder": "qwen35_0.8B_base", "master_weights_on": "gpu", "task_kind": "gpu"},
    "2B": {"folder": "qwen35_2B_base", "master_weights_on": "cpu", "task_kind": "gpu"},
    "4B": {"folder": "qwen35_4B_base", "master_weights_on": "cpu", "task_kind": "gpu"},
    "9B": {"folder": "qwen35_9B_base", "master_weights_on": "cpu", "task_kind": "node"},
}


def check_array_is_orthogonal(array: np.ndarray) -> None:
    """Every pair of columns must show every pair of levels equally often."""
    for first_column, second_column in itertools.combinations(range(array.shape[1]), 2):
        pairs, counts = np.unique(array[:, [first_column, second_column]], axis=0, return_counts=True)
        level_pair_count = len(np.unique(array[:, first_column])) * len(np.unique(array[:, second_column]))
        assert len(pairs) == level_pair_count and len(set(counts)) == 1, (
            f"columns {first_column} and {second_column} of the array are not balanced: {dict(zip(map(tuple, pairs), counts))}")


check_array_is_orthogonal(l18_array)


# ---------------------------------------------------------------------------------------------------------------
# design: from factor levels to the exact shares of every run
# ---------------------------------------------------------------------------------------------------------------
def load_design(design_path: str | Path) -> dict:
    design = yaml.safe_load(Path(design_path).read_text())
    design.setdefault("name", Path(design_path).stem)
    used_columns = [factor["column"] for factor in design["factors"].values()]
    assert len(set(used_columns)) == len(used_columns), f"{design['name']}: two factors share a column"
    for factor_name, factor in design["factors"].items():
        level_count_of_column = len(np.unique(l18_array[:, factor["column"] - 1]))
        assert len(factor["levels"]) == level_count_of_column, (
            f"{design['name']}: {factor_name} has {len(factor['levels'])} levels, column {factor['column']} has {level_count_of_column}")
    for split_name in ("replay_split", "curated_split", "synthetic_split"):
        assert abs(sum(design[split_name].values()) - 1.0) < 1e-9, f"{design['name']}: {split_name} does not sum to 1"
    return design


def shares_of_all_tokens(levels: dict, design: dict) -> dict[str, float]:
    """The share of all training tokens that each component of the law gets, replay included (they sum to 1).

    The factors nest: replay first; a fixed part of the rest goes to long sequences when the run has them; what is
    left is split into rewritten and real text; the real text into hard and curated; the hard text into math and code.
    """
    replay = levels["replay_share"]
    non_replay = 1.0 - replay
    long = design["long_context"]["share_of_non_replay"] * non_replay if levels["long_context"] != "none" else 0.0
    short_non_replay = non_replay - long
    synthetic = levels["synthetic_share_of_non_replay"] * short_non_replay
    real = short_non_replay - synthetic
    hard = levels["hard_share_of_real"] * real
    math = levels["math_share_of_hard"] * hard
    shares = {"replay": replay, "synthetic": synthetic, "curated": real - hard, "math": math, "code": hard - math,
              "long_random": long if levels["long_context"] == "random" else 0.0,
              "long_semantic": long if levels["long_context"] == "semantic" else 0.0}
    assert abs(sum(shares.values()) - 1.0) < 1e-9, f"shares sum to {sum(shares.values())}"
    return shares


def recipe_of_run(run_name: str, levels: dict, design: dict) -> dict:
    """The recipe (midtrain.mixture) of one run. `components` are shares of the ordinary sequences, so they are the
    shares of all tokens divided by what the long sequences leave."""
    shares = shares_of_all_tokens(levels, design)
    long_share = shares["long_random"] + shares["long_semantic"]
    source_shares: dict[str, float] = {}
    splits = {"replay": design["replay_split"], "curated": design["curated_split"], "synthetic": design["synthetic_split"],
              "math": {"math_web": 1.0}, "code": {"code": 1.0}}
    for component_name, split in splits.items():
        for source_name, part in split.items():
            share_of_short_tokens = shares[component_name] * part / (1.0 - long_share)
            if share_of_short_tokens > 0:
                source_shares[source_name] = round(share_of_short_tokens, 6)
    # rounding to six decimals leaves a few millionths over: the largest replay source takes them
    largest_replay_source = max(design["replay_split"], key=design["replay_split"].get)
    source_shares[largest_replay_source] = round(source_shares[largest_replay_source] + 1.0 - sum(source_shares.values()), 6)
    recipe = {"name": run_name, "token_budget": design["token_budget"], "context_length": design["context_length"],
              "components": source_shares, "schedule": levels["schedule"], "seed": design["seed"]}
    if long_share > 0:
        recipe["long_context"] = {"token_share": round(long_share, 6), "sequence_length": design["long_context"]["sequence_length"],
                                  "pack": design["long_context"]["packs"][levels["long_context"]]}
    return recipe


def run_table_of_design(design: dict) -> list[dict]:
    """One entry per run: its name, the level of every factor, the shares of all tokens, the recipe."""
    runs = []
    for run_index, array_row in enumerate(l18_array):
        levels = {factor_name: factor["levels"][array_row[factor["column"] - 1]] for factor_name, factor in design["factors"].items()}
        level_indices = {factor_name: int(array_row[factor["column"] - 1]) for factor_name, factor in design["factors"].items()}
        run_name = f"{design['name']}_r{run_index + 1:02d}"
        runs.append({"run": run_name, "levels": levels, "level_indices": level_indices,
                     "empty_column_levels": {f"empty_column_{column + 1}": int(array_row[column]) for column in range(8)
                                             if column + 1 not in [factor["column"] for factor in design["factors"].values()]},
                     "shares": {name: round(share, 6) for name, share in shares_of_all_tokens(levels, design).items()},
                     "recipe": recipe_of_run(run_name, levels, design)})
    return runs


def write_design_files(design: dict) -> list[dict]:
    """Write one recipe per run under configs/recipes/<design>/ and the table of runs under results/designs/."""
    runs = run_table_of_design(design)
    recipe_directory = repository_root / "configs" / "recipes" / design["name"]
    recipe_directory.mkdir(parents=True, exist_ok=True)
    for run in runs:
        header = (f"# Written by `python -m midtrain.mixing_law design`; do not edit by hand.\n"
                  f"# Factor levels: {json.dumps(run['levels'])}\n# Shares of all tokens: {json.dumps(run['shares'])}\n")
        (recipe_directory / f"{run['run']}.yaml").write_text(header + yaml.safe_dump(run["recipe"], sort_keys=False))
    table_path = repository_root / "results" / "designs" / f"{design['name']}.json"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(json.dumps({"design": design["name"], "factors": design["factors"],
                                      "runs": [{key: run[key] for key in ("run", "levels", "shares")} for run in runs]}, indent=1) + "\n")
    print(f"{design['name']}: {len(runs)} recipes in {recipe_directory.relative_to(repository_root)}, table in {table_path.relative_to(repository_root)}")
    print(f"{'run':>14s} " + " ".join(f"{name[:11]:>11s}" for name in ["lr", "replay", *law_component_names, "schedule"]))
    for run in runs:
        print(f"{run['run']:>14s} {run['levels']['peak_learning_rate']:>11.0e} {run['shares']['replay']:>11.3f} "
              + " ".join(f"{run['shares'][name]:>11.3f}" for name in law_component_names) + f" {run['levels']['schedule']:>11s}")
    return runs


# ---------------------------------------------------------------------------------------------------------------
# queue: one restartable task per run
# ---------------------------------------------------------------------------------------------------------------
def write_queue_tasks(design: dict, model_size: str, queue_directory: str, code_root: str, task_prefix: str,
                      stages: list[str]) -> None:
    """One shell script per run in <queue>/pending. A stage whose result file exists is skipped, so a task that
    failed in its last stage can be queued again without repeating the first ones."""
    size_settings = settings_of_model_size[model_size]
    pending_directory = Path(queue_directory) / "pending"
    pending_directory.mkdir(parents=True, exist_ok=True)
    for run in run_table_of_design(design):
        run_name = f"{run['run']}_{model_size}"
        checkpoint = f"$MIDTRAIN_DATA/runs/{run_name}/checkpoint_final"
        stage_commands = {
            "train": (f"[ -f {checkpoint}/config.json ] || $MIDTRAIN_PYTHON -m midtrain.train_continue "
                      f"--base-model $MIDTRAIN_MODELS/{size_settings['folder']} --recipe configs/recipes/{design['name']}/{run['run']}.yaml "
                      f"--run-name {run_name} --peak-learning-rate {run['levels']['peak_learning_rate']} "
                      f"--master-weights-on {size_settings['master_weights_on']}"),
            "regression": (f"[ -f results/regression/{run_name}.json ] || $MIDTRAIN_PYTHON -m midtrain.base_regression run "
                           f"--checkpoint {checkpoint} --run-name {run_name}"),
            "probe": (f"[ -f results/probe/{run_name}.json ] || $MIDTRAIN_PYTHON -m midtrain.sft_probe run "
                      f"--checkpoint {checkpoint} --run-name {run_name} --master-weights-on {size_settings['master_weights_on']}"),
        }
        lines = ["#!/bin/bash", f"# {design['name']} run {run['run']} on the {model_size} model: " + ", then ".join(stages),
                 f"# factor levels: {json.dumps(run['levels'])}", "set -e", f"cd {code_root} && source scripts/env.sh"]
        lines += [stage_commands[stage] for stage in stages]
        task_path = pending_directory / f"{size_settings['task_kind']}__{task_prefix}{run_name}.sh"
        task_path.write_text("\n".join(lines) + "\n")
        task_path.chmod(0o755)
    print(f"{len(l18_array)} tasks for the {model_size} model in {pending_directory}")


# ---------------------------------------------------------------------------------------------------------------
# analyze: collect the measurements of every run
# ---------------------------------------------------------------------------------------------------------------
def read_json_if_present(file_path: Path) -> dict | None:
    return json.loads(file_path.read_text()) if file_path.exists() else None


def measurements_of_run(run_name: str) -> dict[str, float]:
    """Every number a finished run left under results/, under one flat name each:
    loss/<source> (bits per byte after training), probe/<task>/pass@k, regression/<score name>."""
    measurements: dict[str, float] = {}
    training = read_json_if_present(repository_root / "results" / "training" / f"{run_name}.json")
    if training:
        for source_name, entry in training["validation_after"].items():
            # runs started before 11-20 22:10 also measured the first, discarded batch of rewritten text; later runs do not
            if not source_name.startswith("synthetic_"):
                measurements[f"loss/{source_name}"] = entry["bits_per_byte"]
    probe = read_json_if_present(repository_root / "results" / "probe" / f"{run_name}.json")
    if probe:
        for k in (1, 16, 64):
            values = [task[f"pass@{k}"]["value"] for task in probe["tasks"].values() if f"pass@{k}" in task]
            for task_name, task in probe["tasks"].items():
                if f"pass@{k}" in task:
                    measurements[f"probe/{task_name}/pass@{k}"] = task[f"pass@{k}"]["value"]
            if values:
                measurements[f"probe/mean/pass@{k}"] = float(np.mean(values))
    regression = read_json_if_present(repository_root / "results" / "regression" / f"{run_name}.json")
    if regression:
        scores = regression["scores"]
        for score_name, value in scores.items():
            measurements[f"regression/{score_name}"] = value
        # The guard of the mixture search: every multiple-choice set asked with and without examples, plus trivia.
        # The 0.8B base model picks the letter "B" more often when it has seen five examples (33 of 80 ARC picks
        # against 27 without; accuracy 63.8% against 67.5%; 11-20), so neither format alone is a clean measure of
        # what the model knows; the mean of both uses twice the questions. Chosen before any design run had a result.
        set_names = [name.split("/", 1)[1] for name in scores if name.startswith("multiple_choice/")]
        if set_names and all(f"multiple_choice_zero_shot/{name}" in scores for name in set_names) and "trivia_exact_match" in scores:
            per_set = [(scores[f"multiple_choice/{name}"] + scores[f"multiple_choice_zero_shot/{name}"]) / 2 for name in set_names]
            measurements["regression/knowledge_mean_both_formats"] = float(np.mean(per_set + [scores["trivia_exact_match"]]))
    return measurements


def collect_measurements(design: dict, model_size: str) -> tuple[list[dict], dict[str, np.ndarray]]:
    """The runs of the design that have results, and for every measurement all of them share an array over those runs."""
    runs = run_table_of_design(design)
    per_run = [measurements_of_run(f"{run['run']}_{model_size}") for run in runs]
    finished = [index for index, measurements in enumerate(per_run) if measurements]
    shared_names = sorted(set.intersection(*(set(per_run[index]) for index in finished))) if finished else []
    return ([runs[index] for index in finished],
            {name: np.array([per_run[index][name] for index in finished]) for name in shared_names})


# ---------------------------------------------------------------------------------------------------------------
# analyze: level means, the law, the interference matrix, the best mixture, agreement between sizes
# ---------------------------------------------------------------------------------------------------------------
def level_means_of_factors(runs: list[dict], values: np.ndarray) -> dict:
    """For every factor (and every empty column) the mean measurement at each level, and the range of those means.
    The range of an empty column is what a factor without any effect would show: the yardstick for the others."""
    result = {}
    columns = {**{name: [run["level_indices"][name] for run in runs] for name in runs[0]["level_indices"]},
               **{name: [run["empty_column_levels"][name] for run in runs] for name in runs[0]["empty_column_levels"]}}
    for factor_name, level_of_run in columns.items():
        level_of_run = np.array(level_of_run)
        means = [float(values[level_of_run == level].mean()) for level in sorted(set(level_of_run.tolist()))]
        result[factor_name] = {"level_means": [round(mean, 4) for mean in means], "range": round(max(means) - min(means), 4)}
    return result


def regressors_of_runs(runs: list[dict]) -> tuple[np.ndarray, list[str]]:
    """The design matrix of the law: intercept, one column per component share, two schedule indicators and one
    for the higher learning rate. The intercept is the prediction for replay only, uniform schedule, lower rate."""
    learning_rates = sorted({run["levels"]["peak_learning_rate"] for run in runs})
    names = ["intercept", *law_component_names, "schedule_late", "schedule_early", "higher_learning_rate"]
    rows = [[1.0, *[run["shares"][name] for name in law_component_names],
             float(run["levels"]["schedule"] == "late"), float(run["levels"]["schedule"] == "early"),
             float(run["levels"]["peak_learning_rate"] == learning_rates[-1] and len(learning_rates) > 1)] for run in runs]
    return np.array(rows), names


def fit_mixing_law(runs: list[dict], values: np.ndarray) -> dict:
    """Least squares of one measurement on the regressors. Returns the coefficients with their standard errors, the
    residual standard deviation (the noise of one run) and its degrees of freedom, and the share of the spread over
    the runs that the law explains. That share must be read against what a law of the same size explains of pure
    noise, on average (regressors - 1) / (runs - 1): 9/17 = 0.53 for the screening design."""
    design_matrix, names = regressors_of_runs(runs)
    degrees_of_freedom = len(values) - design_matrix.shape[1]
    assert degrees_of_freedom > 0, f"{len(values)} runs cannot fit {design_matrix.shape[1]} coefficients"
    coefficients, _, rank, _ = np.linalg.lstsq(design_matrix, values, rcond=None)
    assert rank == design_matrix.shape[1], "the regressors are linearly dependent: a component never varies in these runs"
    residuals = values - design_matrix @ coefficients
    residual_variance = float(residuals @ residuals) / degrees_of_freedom
    standard_errors = np.sqrt(np.diag(residual_variance * np.linalg.inv(design_matrix.T @ design_matrix)))
    return {"coefficients": {name: float(value) for name, value in zip(names, coefficients)},
            "standard_errors": {name: float(value) for name, value in zip(names, standard_errors)},
            "residual_standard_deviation": residual_variance ** 0.5, "degrees_of_freedom": degrees_of_freedom,
            "condition_number": float(np.linalg.cond(design_matrix)),
            "spread_over_runs": float(np.std(values, ddof=1)),
            "explained_share": 1.0 - float(residuals @ residuals) / float(((values - values.mean()) ** 2).sum()),
            "explained_share_of_pure_noise": (design_matrix.shape[1] - 1) / (len(values) - 1)}


# Pairs written down before the results (README §13): P8 asks whether a loss ranks the runs as the probe does, P11
# whether the fine-tune of the probe shrinks the differences that few-shot generation shows before it.
loss_and_probe_pairs = [("loss/replay_web", "probe/mean/pass@16"), ("loss/math_web", "probe/math500/pass@16"),
                        ("loss/math_web", "probe/gsm8k/pass@16"), ("loss/code", "probe/mbpp/pass@16"),
                        ("loss/code", "probe/humaneval/pass@16")]
before_and_after_fine_tune_pairs = [("regression/few_shot_mean/pass@1", "probe/mean/pass@1"),
                                    ("regression/few_shot/gsm8k/pass@1", "probe/gsm8k/pass@1"),
                                    ("regression/few_shot/math500/pass@1", "probe/math500/pass@1")]


def prediction_checks(measurements: dict[str, np.ndarray]) -> dict:
    """The rank correlations of P8 and the spreads before and after the fine-tune of P11, over the runs of a design."""
    checks: dict = {"loss_against_probe": {}, "before_and_after_fine_tune": {}}
    for loss_name, probe_name in loss_and_probe_pairs:
        if loss_name in measurements and probe_name in measurements:
            checks["loss_against_probe"][f"{loss_name} vs {probe_name}"] = spearman_with_interval(
                measurements[loss_name], measurements[probe_name])
    for before_name, after_name in before_and_after_fine_tune_pairs:
        if before_name in measurements and after_name in measurements:
            before_spread = float(np.std(measurements[before_name], ddof=1))
            after_spread = float(np.std(measurements[after_name], ddof=1))
            checks["before_and_after_fine_tune"][f"{before_name} vs {after_name}"] = {
                "spread_before": before_spread, "spread_after": after_spread, "ratio": before_spread / after_spread,
                "rank_agreement": spearman_with_interval(measurements[before_name], measurements[after_name])}
    return checks


def interference_matrix(runs: list[dict], measurements: dict[str, np.ndarray], share_step: float = 0.10) -> dict:
    """For every measurement, its change when a component takes `share_step` of all tokens from replay, with the
    ratio of that change to its standard error. Rows are measurements, columns are components."""
    matrix = {}
    for measurement_name, values in measurements.items():
        law = fit_mixing_law(runs, values)
        matrix[measurement_name] = {
            component: {"change": round(share_step * law["coefficients"][component], 4),
                        "change_over_standard_error": round(law["coefficients"][component] / max(1e-12, law["standard_errors"][component]), 2)}
            for component in law_component_names}
    return matrix


def predict_from_law(law: dict, shares: dict, schedule: str, is_higher_learning_rate: bool) -> float:
    coefficients = law["coefficients"]
    return (coefficients["intercept"] + sum(coefficients[name] * shares[name] for name in law_component_names)
            + coefficients["schedule_late"] * (schedule == "late") + coefficients["schedule_early"] * (schedule == "early")
            + coefficients["higher_learning_rate"] * is_higher_learning_rate)


def best_mixture_under_constraint(design: dict, target_law: dict, guard_law: dict | None, guard_tolerance: float,
                                  steps_per_factor: int = 5) -> dict:
    """Search the tested ranges of the factors on a grid for the mixture with the highest predicted target whose
    predicted guard measurement is at most `guard_tolerance` below the replay-only prediction.

    The search never leaves the ranges of the design: the law is a fit, not a theory, and outside the runs that
    produced it nothing supports it. With a linear law the best point lies on the edge of the ranges; the caller
    should read that as "the range was too small on this side" and extend the design there (README §6.4)."""
    factors = design["factors"]
    grids = {}
    for factor_name, factor in factors.items():
        levels = factor["levels"]
        is_numeric = all(isinstance(level, (int, float)) for level in levels)
        grids[factor_name] = (list(np.linspace(min(levels), max(levels), steps_per_factor))
                              if is_numeric and factor_name != "peak_learning_rate" else levels)
    learning_rates = sorted(factors["peak_learning_rate"]["levels"])
    replay_only_guard = guard_law["coefficients"]["intercept"] if guard_law else None
    candidates = []
    for combination in itertools.product(*grids.values()):
        levels = dict(zip(grids, combination))
        shares = shares_of_all_tokens(levels, design)
        is_higher = levels["peak_learning_rate"] == learning_rates[-1] and len(learning_rates) > 1
        predicted_target = predict_from_law(target_law, shares, levels["schedule"], is_higher)
        predicted_guard = predict_from_law(guard_law, shares, levels["schedule"], is_higher) if guard_law else None
        if guard_law and predicted_guard < replay_only_guard - guard_tolerance:
            continue
        candidates.append({"levels": {name: (float(f"{value:.4g}") if isinstance(value, (int, float)) else value)
                                      for name, value in levels.items()},
                           "shares": {name: round(float(share), 4) for name, share in shares.items()},
                           "predicted_target": round(float(predicted_target), 4),
                           "predicted_guard": None if predicted_guard is None else round(float(predicted_guard), 4)})
    assert candidates, "no mixture inside the tested ranges keeps the guard measurement: loosen the tolerance or look at the guard's law"
    candidates.sort(key=lambda candidate: -candidate["predicted_target"])
    best = candidates[0]
    on_edge = [name for name, value in best["levels"].items()
               if isinstance(value, float) and name != "peak_learning_rate"
               and (abs(value - min(factors[name]["levels"])) < 1e-9 or abs(value - max(factors[name]["levels"])) < 1e-9)]
    return {"best": best, "next_best": candidates[1:5], "factors_at_the_edge_of_their_range": on_edge,
            "candidates_within_the_guard": len(candidates)}


def spearman_with_interval(first_values: np.ndarray, second_values: np.ndarray, resample_count: int = 2000) -> dict:
    """Rank correlation of two orderings of the same runs, with a 95% interval from resampling the runs."""
    from scipy.stats import spearmanr

    random_generator = np.random.default_rng(0)
    resampled = []
    for _ in range(resample_count):
        picked = random_generator.integers(0, len(first_values), len(first_values))
        if len(set(first_values[picked])) > 1 and len(set(second_values[picked])) > 1:
            resampled.append(spearmanr(first_values[picked], second_values[picked])[0])
    low, high = np.percentile(resampled, [2.5, 97.5])
    return {"spearman": round(float(spearmanr(first_values, second_values)[0]), 3), "low": round(float(low), 3), "high": round(float(high), 3)}


def analyze_design(design: dict, model_size: str, target_name: str, guard_name: str, guard_tolerance: float,
                   compare_with: str | None) -> dict:
    runs, measurements = collect_measurements(design, model_size)
    assert len(runs) >= 12, f"only {len(runs)} of {len(l18_array)} runs of {design['name']} have results for the {model_size} model"
    analysis = {"design": design["name"], "model_size": model_size, "runs_with_results": len(runs),
                "level_means": {name: level_means_of_factors(runs, values) for name, values in measurements.items()},
                "laws": {name: fit_mixing_law(runs, values) for name, values in measurements.items()},
                "interference_matrix": interference_matrix(runs, measurements),
                "prediction_checks": prediction_checks(measurements)}
    if target_name in measurements:
        guard_law = analysis["laws"].get(guard_name)
        analysis["best_mixture"] = {"target": target_name, "guard": guard_name if guard_law else None, "guard_tolerance": guard_tolerance,
                                    **best_mixture_under_constraint(design, analysis["laws"][target_name], guard_law, guard_tolerance)}
    if compare_with:
        other_runs, other_measurements = collect_measurements(design, compare_with)
        shared_run_names = [run["run"] for run in runs if run["run"] in {other["run"] for other in other_runs}]
        position, other_position = ({run["run"]: index for index, run in enumerate(runs)},
                                    {run["run"]: index for index, run in enumerate(other_runs)})
        agreement = {}
        for name in sorted(set(measurements) & set(other_measurements)):
            first = measurements[name][[position[run_name] for run_name in shared_run_names]]
            second = other_measurements[name][[other_position[run_name] for run_name in shared_run_names]]
            agreement[name] = spearman_with_interval(first, second)
            other_law = fit_mixing_law(other_runs, other_measurements[name])
            agreement[name]["components_with_the_same_sign"] = [
                component for component in law_component_names
                if analysis["laws"][name]["coefficients"][component] * other_law["coefficients"][component] > 0]
        analysis["rank_agreement"] = {"with_model_size": compare_with, "shared_runs": len(shared_run_names), "measurements": agreement}
    output_path = repository_root / "results" / "designs" / f"{design['name']}_{model_size}_analysis.json"
    output_path.write_text(json.dumps(analysis, indent=1) + "\n")
    return analysis


# ---------------------------------------------------------------------------------------------------------------
# compare: two runs, question by question
# ---------------------------------------------------------------------------------------------------------------
def paired_difference_with_interval(first_values: np.ndarray, second_values: np.ndarray, resample_count: int = 2000) -> dict:
    """Mean of (first - second) over the questions, in points, with a 95% interval from resampling the questions.
    The interval excludes 0 when the two runs differ by more than the choice of questions explains."""
    differences = 100.0 * (np.asarray(first_values, dtype=float) - np.asarray(second_values, dtype=float))
    random_generator = np.random.default_rng(0)
    resampled_means = differences[random_generator.integers(0, len(differences), size=(resample_count, len(differences)))].mean(axis=1)
    low, high = np.percentile(resampled_means, [2.5, 97.5])
    return {"difference": round(float(differences.mean()), 2), "low": round(float(low), 2), "high": round(float(high), 2),
            "questions": len(differences)}


def compare_two_runs(first_run_name: str, second_run_name: str) -> dict:
    """First minus second on everything both runs measured per question: pass@1, pass@16 and pass@64 after the
    probe (per task and pooled over the tasks), and the multiple-choice sets of the regression suite."""
    from midtrain.eval_tasks import estimate_pass_at_k

    comparison: dict = {"first": first_run_name, "second": second_run_name, "probe": {}, "regression": {}}
    first_probe = read_json_if_present(repository_root / "results" / "probe" / f"{first_run_name}.json")
    second_probe = read_json_if_present(repository_root / "results" / "probe" / f"{second_run_name}.json")
    if first_probe and second_probe:
        pooled: dict[int, tuple[list, list]] = {}
        for task_name in sorted(set(first_probe["tasks"]) & set(second_probe["tasks"])):
            first_task, second_task = first_probe["tasks"][task_name], second_probe["tasks"][task_name]
            assert first_task["item_ids"] == second_task["item_ids"], f"{task_name}: the two runs answered different questions"
            sample_count = first_task.get("sample_count", first_probe.get("sample_count"))
            assert sample_count == second_task.get("sample_count", second_probe.get("sample_count")), f"{task_name}: different sample counts"
            comparison["probe"][task_name] = {}
            for k in (1, 16, 64):
                if k > sample_count:
                    continue
                first_values = [estimate_pass_at_k(correct, sample_count, k) for correct in first_task["correct_counts"]]
                second_values = [estimate_pass_at_k(correct, sample_count, k) for correct in second_task["correct_counts"]]
                comparison["probe"][task_name][f"pass@{k}"] = paired_difference_with_interval(first_values, second_values)
                pooled.setdefault(k, ([], []))
                pooled[k][0].extend(first_values)
                pooled[k][1].extend(second_values)
        comparison["probe"]["all_tasks"] = {f"pass@{k}": paired_difference_with_interval(first_values, second_values)
                                            for k, (first_values, second_values) in pooled.items()}
    first_regression = read_json_if_present(repository_root / "results" / "regression" / f"{first_run_name}.json")
    second_regression = read_json_if_present(repository_root / "results" / "regression" / f"{second_run_name}.json")
    if first_regression and second_regression:
        pooled_first, pooled_second = [], []
        for set_name in sorted(set(first_regression["item_correct"]) & set(second_regression["item_correct"])):
            first_values, second_values = first_regression["item_correct"][set_name], second_regression["item_correct"][set_name]
            comparison["regression"][set_name] = paired_difference_with_interval(first_values, second_values)
            if set_name.startswith("multiple_choice/"):
                pooled_first.extend(first_values)
                pooled_second.extend(second_values)
        comparison["regression"]["multiple_choice/all_sets"] = paired_difference_with_interval(pooled_first, pooled_second)
        comparison["regression"]["scores"] = {name: round(first_regression["scores"][name] - second_regression["scores"][name], 4)
                                              for name in sorted(set(first_regression["scores"]) & set(second_regression["scores"]))}
    output_directory = repository_root / "results" / "comparisons"
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / f"{first_run_name}__minus__{second_run_name}.json").write_text(json.dumps(comparison, indent=1) + "\n")
    return comparison


def compare_runs_with_their_base(model_size: str, without_items_matched_by: str | None = "synthetic_solutions") -> dict:
    """Every run of one model size that has a greedy final evaluation, minus the base model of that size, question
    by question: greedy accuracy on each test set (results/final_eval), pass@1 and pass@16 after the generic probe
    (results/probe_generic), the knowledge guard, and the change of the held-out loss on each source.

    The tables of README §1.2 are read from the file this writes, results/comparisons/dose_vs_base_<size>.json, so
    that every number there traces to one command. A run is left out of a block when it or the base lacks that
    result; a test set is compared only on questions both answered. Each greedy difference is also given without the
    test items that `without_items_matched_by` touches (`data_hygiene.py matched-items`; README §1.2), as
    `without_matched_items`, when that file exists.
    """
    matched_item_ids: set = set()
    if without_items_matched_by:
        matched_path = data_root() / "hygiene" / f"matched_items_{without_items_matched_by}.json"
        if matched_path.exists():
            matched_item_ids = {item_id for item_ids in json.loads(matched_path.read_text()).values() for item_id in item_ids}
    from midtrain.eval_tasks import estimate_pass_at_k

    base_name = f"base_{model_size}"
    folders = {block: repository_root / "results" / block for block in ("final_eval", "probe_generic", "regression", "training")}
    base_final = read_json_if_present(folders["final_eval"] / f"{base_name}.json")
    base_probe = read_json_if_present(folders["probe_generic"] / f"{base_name}.json")
    base_regression = read_json_if_present(folders["regression"] / f"{base_name}.json")
    assert base_final, f"no greedy final evaluation of {base_name}"
    # a run of this size ends in the size (dose_distill_lr3e-6_0.8B) or carries it before a suffix (..._0.8B_merge25)
    run_names = sorted(path.stem for path in folders["final_eval"].glob(f"*_{model_size}*.json")
                       if path.stem != base_name and (path.stem.endswith(f"_{model_size}") or f"_{model_size}_" in path.stem))
    comparison = {"model_size": model_size, "base": base_name, "matched_items_left_out_by": without_items_matched_by,
                  "runs": {}}
    for run_name in run_names:
        run_result = {"few_shot_greedy": {}, "generic_probe": {}}
        run_final = read_json_if_present(folders["final_eval"] / f"{run_name}.json")
        for task_name, task in run_final["tasks"].items():
            if task_name not in base_final["tasks"]:
                continue
            base_task = base_final["tasks"][task_name]
            assert task["item_ids"] == base_task["item_ids"], f"{run_name} {task_name}: different questions"
            run_result["few_shot_greedy"][task_name] = {
                "run": task["pass@1"]["value"], "base": base_task["pass@1"]["value"],
                **paired_difference_with_interval(task["correct_counts"], base_task["correct_counts"])}
            kept_positions = [position for position, item_id in enumerate(task["item_ids"]) if item_id not in matched_item_ids]
            if matched_item_ids and len(kept_positions) < len(task["item_ids"]):
                run_result["few_shot_greedy"][task_name]["without_matched_items"] = paired_difference_with_interval(
                    np.asarray(task["correct_counts"])[kept_positions], np.asarray(base_task["correct_counts"])[kept_positions])
        run_probe = read_json_if_present(folders["probe_generic"] / f"{run_name}.json")
        if run_probe and base_probe:
            for task_name, task in run_probe["tasks"].items():
                base_task = base_probe["tasks"][task_name]
                assert task["item_ids"] == base_task["item_ids"], f"{run_name} {task_name}: different probe questions"
                sample_count = task["sample_count"]
                run_result["generic_probe"][task_name] = {
                    f"pass@{k}": paired_difference_with_interval(
                        [estimate_pass_at_k(count, sample_count, k) for count in task["correct_counts"]],
                        [estimate_pass_at_k(count, sample_count, k) for count in base_task["correct_counts"]])
                    for k in (1, 16) if k <= sample_count}
        run_regression = read_json_if_present(folders["regression"] / f"{run_name}.json")
        if run_regression and base_regression:
            run_result["knowledge_cloze_change"] = round(run_regression["scores"]["knowledge_cloze_mean"]
                                                         - base_regression["scores"]["knowledge_cloze_mean"], 2)
        training = read_json_if_present(folders["training"] / f"{run_name}.json")
        if training:
            run_result["loss_change_percent"] = {
                source: round(100 * (after["bits_per_byte"] - training["validation_before"][source]["bits_per_byte"])
                              / training["validation_before"][source]["bits_per_byte"], 2)
                for source, after in training["validation_after"].items()}
            run_result["tokens_seen"] = training["tokens_seen"]
        comparison["runs"][run_name] = run_result
    output_path = repository_root / "results" / "comparisons" / f"dose_vs_base_{model_size}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(comparison, indent=1) + "\n")
    return comparison


def print_dose_comparison(comparison: dict) -> None:
    for run_name, run_result in comparison["runs"].items():
        few_shot = "  ".join(f"{task} {value['difference']:+.1f} [{value['low']:+.1f}, {value['high']:+.1f}]"
                             for task, value in run_result["few_shot_greedy"].items())
        probe = "  ".join(f"{task} {value['pass@1']['difference']:+.1f}" for task, value in run_result["generic_probe"].items())
        print(f"{run_name}\n  greedy few-shot: {few_shot}\n  generic probe pass@1: {probe or '-'}"
              f"\n  knowledge: {run_result.get('knowledge_cloze_change', '-')}")


def print_comparison(comparison: dict) -> None:
    print(f"{comparison['first']}  minus  {comparison['second']}  (points, 95% interval over the questions)")
    for task_name, entries in comparison["probe"].items():
        print(f"  probe {task_name:12s} " + "   ".join(
            f"{name} {entry['difference']:+6.2f} [{entry['low']:+.2f}, {entry['high']:+.2f}]" for name, entry in entries.items()))
    for set_name, entry in comparison["regression"].items():
        if set_name != "scores":
            print(f"  {set_name:44s} {entry['difference']:+6.2f} [{entry['low']:+.2f}, {entry['high']:+.2f}]  ({entry['questions']} questions)")
    for name, difference in comparison["regression"].get("scores", {}).items():
        if name.startswith(("few_shot", "context", "trivia", "shot_gain_mean", "knowledge")):
            print(f"  {name:44s} {difference:+.3f}")


def print_analysis(analysis: dict, shown_measurements: list[str]) -> None:
    for name in shown_measurements:
        if name not in analysis["laws"]:
            continue
        law = analysis["laws"][name]
        print(f"\n{name}: noise of one run {law['residual_standard_deviation']:.3f} ({law['degrees_of_freedom']} degrees of freedom)")
        for factor_name, entry in analysis["level_means"][name].items():
            print(f"  {factor_name:32s} level means {entry['level_means']}  range {entry['range']}")
        print("  per +10% of all tokens taken from replay: " + ", ".join(
            f"{component} {values['change']:+.3f} ({values['change_over_standard_error']:+.1f} se)"
            for component, values in analysis["interference_matrix"][name].items()))
    if "best_mixture" in analysis:
        best = analysis["best_mixture"]
        print(f"\nbest mixture for {best['target']} with {best['guard']} within {best['guard_tolerance']}: "
              f"{json.dumps(best['best'])}\n  at the edge of the tested range: {best['factors_at_the_edge_of_their_range']}")
    if "rank_agreement" in analysis:
        print(f"\nrank agreement with the {analysis['rank_agreement']['with_model_size']} model over {analysis['rank_agreement']['shared_runs']} runs:")
        for name in shown_measurements:
            if name in analysis["rank_agreement"]["measurements"]:
                print(f"  {name:32s} {json.dumps(analysis['rank_agreement']['measurements'][name])}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    design_parser = commands.add_parser("design")
    design_parser.add_argument("design")
    queue_parser = commands.add_parser("queue")
    queue_parser.add_argument("design")
    queue_parser.add_argument("--model-size", required=True, choices=list(settings_of_model_size))
    queue_parser.add_argument("--queue", required=True, help="the queue folder (scripts/cluster/node_worker.sh)")
    queue_parser.add_argument("--code-root", default=str(repository_root), help="where the repository lives on the machines that run the queue")
    queue_parser.add_argument("--task-prefix", default="", help="text put before the run name: tasks are claimed in the order of their names")
    queue_parser.add_argument("--stages", nargs="+", default=["train", "regression", "probe"], choices=["train", "regression", "probe"])
    analyze_parser = commands.add_parser("analyze")
    analyze_parser.add_argument("design")
    analyze_parser.add_argument("--model-size", required=True)
    analyze_parser.add_argument("--target", default="probe/mean/pass@16")
    analyze_parser.add_argument("--guard", default="regression/knowledge_cloze_mean")
    analyze_parser.add_argument("--guard-tolerance", type=float, default=0.5,
                                help="how far below the replay-only prediction the guard may fall, in its own unit")
    analyze_parser.add_argument("--compare-with", default=None, help="a second model size, for rank agreement")
    analyze_parser.add_argument("--show", nargs="+", default=["probe/mean/pass@1", "probe/mean/pass@16", "probe/mean/pass@64",
                                                             "regression/knowledge_cloze_mean", "loss/math_web", "loss/code",
                                                             "loss/replay_web"])
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("first_run")
    compare_parser.add_argument("second_run")
    dose_parser = commands.add_parser("compare-with-base", help="every run of a size minus its base: README §1.2")
    dose_parser.add_argument("--model-size", required=True)
    arguments = parser.parse_args()
    if arguments.command == "compare":
        print_comparison(compare_two_runs(arguments.first_run, arguments.second_run))
        return
    if arguments.command == "compare-with-base":
        print_dose_comparison(compare_runs_with_their_base(arguments.model_size))
        return
    design = load_design(arguments.design)
    if arguments.command == "design":
        write_design_files(design)
    elif arguments.command == "queue":
        write_queue_tasks(design, arguments.model_size, arguments.queue, arguments.code_root, arguments.task_prefix, arguments.stages)
    else:
        analysis = analyze_design(design, arguments.model_size, arguments.target, arguments.guard, arguments.guard_tolerance,
                                  arguments.compare_with)
        print_analysis(analysis, arguments.show)


if __name__ == "__main__":
    main()
