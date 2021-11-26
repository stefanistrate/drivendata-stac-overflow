"""Utilities for processing inputs, shared between training and inference."""

from __future__ import annotations


def parse_channels_and_rewrites(
        channel_specs: list[str]
) -> tuple[list[str], dict[str, tuple[int, int]]]:
    keys = []
    rewrite_map = {}

    for spec in channel_specs:
        parts = spec.split(":")
        if not parts:
            continue
        keys.append(parts[0])
        if len(parts) < 3:
            continue
        rewrite_map[parts[0]] = int(parts[1]), int(parts[2])

    return keys, rewrite_map
