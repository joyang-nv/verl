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

"""Validation-triggered best-checkpoint selection and persisted state."""

import fnmatch
import json
import math
import os
import shutil
import uuid
from collections.abc import Callable, Mapping
from typing import Any

BEST_CHECKPOINT_METADATA = "best_checkpoint.json"


def find_matching_metric(metrics: Mapping[str, Any], metric_pattern: str) -> tuple[str, float]:
    """Resolve an exact metric name or glob pattern to one finite scalar metric."""
    matches = sorted(name for name in metrics if fnmatch.fnmatchcase(name, metric_pattern))
    if not matches:
        raise KeyError(
            f"Best-checkpoint metric pattern {metric_pattern!r} matched no validation metrics. "
            f"Available metrics: {sorted(metrics)}"
        )
    if len(matches) > 1:
        raise ValueError(f"Best-checkpoint metric pattern {metric_pattern!r} is ambiguous; matched metrics: {matches}")

    metric_name = matches[0]
    try:
        metric_value = float(metrics[metric_name])
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Best-checkpoint metric {metric_name!r} must be a scalar number") from exc
    if not math.isfinite(metric_value):
        raise ValueError(f"Best-checkpoint metric {metric_name!r} must be finite, got {metric_value}")
    return metric_name, metric_value


def load_best_checkpoint_metadata(best_checkpoint_dir: str) -> dict[str, Any] | None:
    """Load the persisted best-checkpoint state, if present."""
    metadata_path = os.path.join(best_checkpoint_dir, BEST_CHECKPOINT_METADATA)
    if not os.path.isfile(metadata_path):
        return None
    with open(metadata_path) as metadata_file:
        metadata = json.load(metadata_file)

    required_keys = {"metric_pattern", "metric_name", "metric_value", "global_step", "checkpoint_path"}
    missing_keys = required_keys.difference(metadata)
    if missing_keys:
        raise ValueError(f"Invalid best-checkpoint metadata at {metadata_path}: missing {sorted(missing_keys)}")
    return metadata


def should_save_best_checkpoint(metric_value: float, metadata: Mapping[str, Any] | None, metric_pattern: str) -> bool:
    """Return whether a metric is better than the persisted maximum."""
    if metadata is None or metadata.get("metric_pattern") != metric_pattern:
        return True
    return metric_value > float(metadata["metric_value"])


def _atomic_write_json(path: str, value: Mapping[str, Any]) -> None:
    temporary_path = f"{path}.tmp-{uuid.uuid4().hex}"
    try:
        with open(temporary_path, "w") as temporary_file:
            json.dump(value, temporary_file, indent=2, sort_keys=True)
            temporary_file.write("\n")
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def maybe_save_best_checkpoint(
    metrics: Mapping[str, Any],
    metric_pattern: str,
    best_checkpoint_dir: str,
    global_step: int,
    save_checkpoint: Callable[[str], None],
) -> dict[str, Any] | None:
    """Trigger a checkpoint save when validation produces a new maximum metric."""
    metric_name, metric_value = find_matching_metric(metrics, metric_pattern)
    best_checkpoint_dir = os.path.abspath(best_checkpoint_dir)
    previous_metadata = load_best_checkpoint_metadata(best_checkpoint_dir)
    if not should_save_best_checkpoint(metric_value, previous_metadata, metric_pattern):
        return None

    save_checkpoint(best_checkpoint_dir)
    checkpoint_path = os.path.join(best_checkpoint_dir, f"global_step_{global_step}")
    if not os.path.isdir(checkpoint_path):
        raise RuntimeError(f"Best-checkpoint save did not create the expected directory: {checkpoint_path}")

    metadata = {
        "metric_pattern": metric_pattern,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "global_step": global_step,
        "checkpoint_path": checkpoint_path,
    }
    os.makedirs(best_checkpoint_dir, exist_ok=True)
    _atomic_write_json(os.path.join(best_checkpoint_dir, BEST_CHECKPOINT_METADATA), metadata)

    if previous_metadata is not None and int(previous_metadata["global_step"]) != global_step:
        previous_checkpoint = os.path.join(best_checkpoint_dir, f"global_step_{int(previous_metadata['global_step'])}")
        if os.path.isdir(previous_checkpoint):
            shutil.rmtree(previous_checkpoint)

    return metadata
