"""Tests of the regression suite's final evaluation (base_regression.py) that need no GPU; seconds.

Why these tests exist. The final evaluation gained two sets on 11-23 (SVAMP, ASDiv-A). Running it again on an
earlier run must add the new sets and leave the sets already measured untouched, so that a number in the README
never changes because a later set was added.

    source scripts/env.sh && $MIDTRAIN_PYTHON -m pytest tests/test_base_regression.py -q
"""
import json

from midtrain import base_regression


def test_final_evaluation_fills_in_only_the_missing_sets(monkeypatch, tmp_path):
    requested_task_names = []

    def pretend_to_evaluate(checkpoint_path, run_name, port, prompt_style, evaluation_settings):
        requested_task_names.extend(evaluation_settings["tasks"])
        return {"tasks": {task_name: {"pass@1": {"value": 50.0}, "item_ids": ["a", "b"], "correct_counts": [1, 0]}
                          for task_name in evaluation_settings["tasks"]}}

    monkeypatch.setattr(base_regression, "repository_root", tmp_path)
    monkeypatch.setattr(base_regression.sft_probe, "evaluate_pass_at_k", pretend_to_evaluate)
    monkeypatch.setattr(base_regression.sft_probe, "port_for_this_process", lambda: 8100)
    result_path = tmp_path / "results" / "final_eval" / "some_run.json"
    result_path.parent.mkdir(parents=True)
    measured_before = {"pass@1": {"value": 42.7}, "item_ids": ["x"], "correct_counts": [1]}
    result_path.write_text(json.dumps({"run_name": "some_run", "decoding": "greedy, one answer per item",
                                       "tasks": {"gsm8k": measured_before}}))

    summary = base_regression.run_final_evaluation("unused", "some_run")
    every_set = list(base_regression.final_generation_settings["tasks"])
    assert requested_task_names == [task_name for task_name in every_set if task_name != "gsm8k"]
    stored = json.loads(result_path.read_text())["tasks"]
    assert stored["gsm8k"] == measured_before and set(stored) == set(every_set) and summary["gsm8k"]["accuracy"] == 42.7

    requested_task_names.clear()
    base_regression.run_final_evaluation("unused", "some_run")
    assert requested_task_names == []          # nothing missing: no server is started at all
