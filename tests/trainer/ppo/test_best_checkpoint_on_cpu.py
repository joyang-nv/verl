# Copyright 2026 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

import pytest

from verl.trainer.ppo.best_checkpoint import (
    BEST_CHECKPOINT_METADATA,
    find_matching_metric,
    load_best_checkpoint_metadata,
    maybe_save_best_checkpoint,
)


def test_find_matching_metric_supports_exact_name_and_glob():
    metrics = {
        "val-core/aime24/acc/mean@1": 0.5,
        "val-aux/aime24/reward/mean@1": 0.4,
    }

    assert find_matching_metric(metrics, "val-core/aime24/acc/mean@1") == (
        "val-core/aime24/acc/mean@1",
        0.5,
    )
    assert find_matching_metric(metrics, "val-core/*/acc/mean@1") == (
        "val-core/aime24/acc/mean@1",
        0.5,
    )


def test_find_matching_metric_rejects_missing_or_ambiguous_patterns():
    metrics = {
        "val-core/aime24/acc/mean@1": 0.5,
        "val-core/aime25/acc/mean@1": 0.6,
    }

    with pytest.raises(KeyError, match="matched no validation metrics"):
        find_matching_metric(metrics, "val-core/*/reward/mean@1")
    with pytest.raises(ValueError, match="ambiguous"):
        find_matching_metric(metrics, "val-core/*/acc/mean@1")


def test_validation_improvement_triggers_save_and_persists_across_calls(tmp_path):
    best_checkpoint_dir = tmp_path / "best"
    save_calls = []

    def save_checkpoint(checkpoint_dir: str):
        step = len(save_calls) * 10 + 10
        save_calls.append(checkpoint_dir)
        checkpoint = best_checkpoint_dir / f"global_step_{step}" / "actor"
        checkpoint.mkdir(parents=True)
        (checkpoint / "model.pt").write_text(f"step-{step}")

    metric_pattern = "val-core/*/acc/mean@1"
    metric_name = "val-core/aime24/acc/mean@1"

    first = maybe_save_best_checkpoint(
        {metric_name: 0.4},
        metric_pattern,
        best_checkpoint_dir,
        global_step=10,
        save_checkpoint=save_checkpoint,
    )
    assert len(save_calls) == 1
    assert first["metric_value"] == 0.4
    assert (best_checkpoint_dir / "global_step_10").is_dir()

    not_better = maybe_save_best_checkpoint(
        {metric_name: 0.4},
        metric_pattern,
        best_checkpoint_dir,
        global_step=20,
        save_checkpoint=save_checkpoint,
    )
    assert not_better is None
    assert len(save_calls) == 1

    second = maybe_save_best_checkpoint(
        {metric_name: 0.6},
        metric_pattern,
        best_checkpoint_dir,
        global_step=20,
        save_checkpoint=save_checkpoint,
    )
    assert len(save_calls) == 2
    assert not (best_checkpoint_dir / "global_step_10").exists()
    assert (best_checkpoint_dir / "global_step_20" / "actor" / "model.pt").read_text() == "step-20"
    assert load_best_checkpoint_metadata(best_checkpoint_dir) == second
    assert json.loads((best_checkpoint_dir / BEST_CHECKPOINT_METADATA).read_text()) == second


def test_failed_improvement_save_keeps_previous_best(tmp_path):
    best_checkpoint_dir = tmp_path / "best"
    metric_pattern = "val-core/*/acc/mean@1"
    metric_name = "val-core/aime24/acc/mean@1"

    def save_first_checkpoint(checkpoint_dir: str):
        checkpoint = best_checkpoint_dir / "global_step_10" / "actor"
        checkpoint.mkdir(parents=True)

    first = maybe_save_best_checkpoint(
        {metric_name: 0.4},
        metric_pattern,
        best_checkpoint_dir,
        global_step=10,
        save_checkpoint=save_first_checkpoint,
    )

    def fail_to_save_checkpoint(checkpoint_dir: str):
        raise RuntimeError("checkpoint write failed")

    with pytest.raises(RuntimeError, match="checkpoint write failed"):
        maybe_save_best_checkpoint(
            {metric_name: 0.6},
            metric_pattern,
            best_checkpoint_dir,
            global_step=20,
            save_checkpoint=fail_to_save_checkpoint,
        )

    assert (best_checkpoint_dir / "global_step_10").is_dir()
    assert load_best_checkpoint_metadata(best_checkpoint_dir) == first
