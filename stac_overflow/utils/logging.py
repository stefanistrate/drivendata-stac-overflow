"""Logging utilities."""

from __future__ import annotations

import apache_beam as beam
import collections.abc
import json
import tensorflow as tf

from typing import Any


def dump(obj: Any) -> str:
    """Dumps an object to a JSON formatted string."""

    return json.dumps(obj, indent=2)


def sg_pl(obj: Any, singular: str, plural: str) -> str:
    """Generates the singular or plural form based on an object cardinality."""

    if isinstance(obj, int):
        cardinality = obj
    elif isinstance(obj, float):
        cardinality = obj
    elif isinstance(obj, collections.abc.Sized):
        cardinality = len(obj)
    else:
        raise TypeError(f"Invalid object type `{type(obj)}`.")

    form = singular if cardinality in {-1, 1} else plural
    return f"{cardinality} {form}"


def print_pipeline_result(result: beam.runners.runner.PipelineResult,
                          print_fn=None) -> None:
    """Prints Beam pipeline result."""

    metrics = []
    for metric_results in result.metrics().query().values():
        for metric_result in metric_results:
            metrics.append({
                "key": str(metric_result.key),
                "committed": str(metric_result.committed),
                "attempted": str(metric_result.attempted),
            })

    if print_fn is None:
        print_fn = print
    print_fn(dump(metrics))


def print_model_summary(model: tf.keras.Model, print_fn=None) -> None:
    """Prints model summary recursively."""

    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            print_model_summary(layer, print_fn=print_fn)
    model.summary(line_length=160, print_fn=print_fn)
