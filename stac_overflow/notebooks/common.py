"""Common utilities."""

import numpy as np
import pandas as pd
import rasterio

from matplotlib.colors import Normalize

from stac_overflow.utils.colormaps import (CM_JRC_CHANGE, CM_JRC_EXTENT,
                                           CM_JRC_OCCURRENCE, CM_JRC_RECURRENCE,
                                           CM_JRC_SEASONALITY,
                                           CM_JRC_TRANSITIONS, CM_WATER_LABEL)

NORM_UINT8 = Normalize(vmin=0, vmax=255)


def load_metadata(datastore):
    metadata = pd.read_csv(datastore / "flood-training-metadata.csv",
                           parse_dates=["scene_start"])
    metadata["feature_path"] = (f"{datastore}/train_features/" +
                                metadata.image_id + ".tif")
    metadata["label_path"] = (f"{datastore}/train_labels/" + metadata.chip_id +
                              ".tif")
    metadata["nasadem_path"] = (f"{datastore}/train_features/" +
                                metadata.chip_id + "_nasadem.tif")
    metadata["jrc_change_path"] = (f"{datastore}/train_features/" +
                                   metadata.chip_id + "_jrc-gsw-change.tif")
    metadata["jrc_extent_path"] = (f"{datastore}/train_features/" +
                                   metadata.chip_id + "_jrc-gsw-extent.tif")
    metadata["jrc_occurrence_path"] = (f"{datastore}/train_features/" +
                                       metadata.chip_id +
                                       "_jrc-gsw-occurrence.tif")
    metadata["jrc_recurrence_path"] = (f"{datastore}/train_features/" +
                                       metadata.chip_id +
                                       "_jrc-gsw-recurrence.tif")
    metadata["jrc_seasonality_path"] = (f"{datastore}/train_features/" +
                                        metadata.chip_id +
                                        "_jrc-gsw-seasonality.tif")
    metadata["jrc_transitions_path"] = (f"{datastore}/train_features/" +
                                        metadata.chip_id +
                                        "_jrc-gsw-transitions.tif")
    return metadata


def load_image_bands(chip_df):
    paths = [
        ("vv", chip_df[chip_df.polarization == "vv"].feature_path.values[0],
         "coolwarm", None),
        ("vh", chip_df[chip_df.polarization == "vh"].feature_path.values[0],
         "coolwarm", None),
        ("nasadem", chip_df.nasadem_path.values[0], "terrain", None),
        ("jrc_change", chip_df.jrc_change_path.values[0], CM_JRC_CHANGE,
         NORM_UINT8),
        ("jrc_extent", chip_df.jrc_extent_path.values[0], CM_JRC_EXTENT,
         NORM_UINT8),
        ("jrc_occurrence", chip_df.jrc_occurrence_path.values[0],
         CM_JRC_OCCURRENCE, NORM_UINT8),
        ("jrc_recurrence", chip_df.jrc_recurrence_path.values[0],
         CM_JRC_RECURRENCE, NORM_UINT8),
        ("jrc_seasonality", chip_df.jrc_seasonality_path.values[0],
         CM_JRC_SEASONALITY, NORM_UINT8),
        ("jrc_transitions", chip_df.jrc_transitions_path.values[0],
         CM_JRC_TRANSITIONS, NORM_UINT8),
    ]

    bands = []
    names = []
    cmaps = []
    norms = []
    for name, path, cmap, norm in paths:
        with rasterio.open(path) as f:
            band = f.read(1)
            bands.append(band)
        names.append(name)
        cmaps.append(cmap)
        norms.append(norm)

    img = np.ma.stack(bands, axis=-1)

    return img, bands, names, cmaps, norms


def load_image_label(chip_df):
    with rasterio.open(chip_df.label_path.values[0]) as f:
        label = f.read(1)
        label = np.ma.masked_equal(label, 255)
    return label, "label", CM_WATER_LABEL, NORM_UINT8


def augment_data(img, label, transforms):
    data = {"image": img, "mask": label}
    augmented = transforms(**data)

    augmented_bands = [
        np.squeeze(band, axis=-1)
        for band in np.split(augmented["image"], img.shape[-1], axis=-1)
    ]

    return augmented["image"], augmented_bands, augmented["mask"]
