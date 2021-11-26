"""Inference for the STAC Overflow task.

See competition at:
https://www.drivendata.org/competitions/81/detect-flood-water/page/385/

This script will average predictions from multiple models, but we have to pay
attention to providing the right inputs (i.e. the correct bands with the
expected rewrites) to each of them.
"""

from __future__ import annotations

import collections
import numpy as np
import rasterio
import tensorflow as tf

from absl import app
from absl import logging
from pathlib import Path
from tensorflow.python.ops.numpy_ops import np_config
from tifffile import imwrite
from tqdm import tqdm

ROOT_DIR = Path("/codeexecution")
SUBMISSION_DIR = ROOT_DIR / "submission"
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"
INPUT_DIR = DATA_DIR / "test_features"
JRC_OCCURRENCE_DIR = DATA_DIR / "jrc_occurrence"
NASADEM_DIR = DATA_DIR / "nasadem"


def spill_water(water: np.ndarray,
                elevation: np.ndarray,
                delta_elevation: int = 25) -> np.ndarray:
    num_rows, num_cols = water.shape
    spilled = np.zeros_like(water)

    def _should_visit(from_row, from_col, to_row, to_col):
        if spilled[to_row, to_col]:
            return False
        return (elevation[from_row, from_col] >=
                elevation[to_row, to_col] + delta_elevation)

    queue = collections.deque([tuple(pos) for pos in np.argwhere(water == 1)])
    try:
        while True:
            row, col = queue.pop()
            spilled[row, col] = 1
            if row > 0 and _should_visit(row, col, row - 1, col):
                queue.append((row - 1, col))
            if row + 1 < num_rows and _should_visit(row, col, row + 1, col):
                queue.append((row + 1, col))
            if col > 0 and _should_visit(row, col, row, col - 1):
                queue.append((row, col - 1))
            if col + 1 < num_cols and _should_visit(row, col, row, col + 1):
                queue.append((row, col + 1))
    except IndexError:
        pass

    return spilled


def list_chip_ids() -> list[str]:
    """Lists chip ids from an input directory."""

    paths = INPUT_DIR.glob("*.tif")
    chip_ids = {path.stem.split("_")[0] for path in paths}
    return list(sorted(chip_ids))


def make_prediction(models: list[tf.keras.Model], chip_id: str) -> np.ndarray:
    """Makes prediction for a given chip."""

    vv_path = INPUT_DIR / f"{chip_id}_vv.tif"
    with rasterio.open(vv_path) as f:
        vv = f.read(1)
    vh_path = INPUT_DIR / f"{chip_id}_vh.tif"
    with rasterio.open(vh_path) as f:
        vh = f.read(1)
    raster = np.stack([vv, vh], axis=-1)
    raster = np.expand_dims(raster, axis=0)

    predictions = [model(raster, training=False) for model in models]
    result = np.mean(predictions, axis=0)
    result = np.squeeze(result)
    result = np.rint(result)
    return result


def postprocess_prediction(pred: np.ndarray, chip_id: str) -> np.ndarray:
    """Postprocesses prediction for a given chip."""

    jrc_occurrence_path = JRC_OCCURRENCE_DIR / f"{chip_id}.tif"
    with rasterio.open(jrc_occurrence_path) as f:
        jrc_occurrence = f.read(1)
    pred[jrc_occurrence == 100] = 1

    nasadem_path = NASADEM_DIR / f"{chip_id}.tif"
    with rasterio.open(nasadem_path) as f:
        nasadem = f.read(1)
    pred = spill_water(pred, nasadem)

    return pred


def main(argv):
    del argv  # Unused.

    np_config.enable_numpy_behavior()

    logging.info("Loading model.")
    models = [
        tf.keras.models.load_model(path, compile=False)
        for path in MODELS_DIR.glob("*.h5")
    ]

    logging.info("Finding all chips.")
    chip_ids = list_chip_ids()
    if not chip_ids:
        raise OSError("No input images found!")
    logging.info("Found %d test chip ids.", len(chip_ids))

    logging.info("Generating predictions.")
    for chip_id in tqdm(chip_ids, miniters=25):
        output_data = make_prediction(models, chip_id).astype(np.uint8)
        output_data = postprocess_prediction(output_data, chip_id)
        output_path = SUBMISSION_DIR / f"{chip_id}.tif"
        imwrite(output_path, output_data, dtype=np.uint8)
    logging.info("Inference complete.")


if __name__ == "__main__":
    app.run(main)
