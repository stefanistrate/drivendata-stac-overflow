"""Datasets utilities."""

from __future__ import annotations

import albumentations
import numpy as np
import tensorflow as tf

from pathlib import PurePath
from rasterio.io import MemoryFile


def augment_image_dataset(dataset: tf.data.Dataset,
                          transforms: albumentations.Compose,
                          augment_labels: bool = False) -> tf.data.Dataset:
    """Augments a non-batched dataset using the given transforms."""

    def _augment(image, label):
        if augment_labels:
            data = {"image": image, "mask": label}
            augmented = transforms(**data)
            return augmented["image"], augmented["mask"]
        else:
            data = {"image": image}
            augmented = transforms(**data)
            return augmented["image"], label

    def _augment_tensors(image_tensor, label_tensor):
        return tf.numpy_function(func=_augment,
                                 inp=[image_tensor, label_tensor],
                                 Tout=[image_tensor.dtype, label_tensor.dtype])

    return dataset.map(_augment_tensors,
                       num_parallel_calls=tf.data.AUTOTUNE,
                       deterministic=True)


def tfrecords_as_geospatial_dataset(
        file_pattern: PurePath = None,
        batch_size: int = 1,
        repeat: bool = False,
        shuffle_buffer_size: int = 0,
        prefetch_buffer_size: int = tf.data.AUTOTUNE,
        transforms: albumentations.Compose = None,
        tfr_channel_keys: list[str] = None,
        tfr_channel_rewrite_map: dict[str, tuple[int, int]] = None,
        tfr_label_key: str = "label") -> tf.data.Dataset:
    """Builds a geospatial raster dataset from TFRecords."""

    if not file_pattern:
        raise ValueError("Must provide a non-empty file pattern.")
    if not tfr_channel_keys:
        raise ValueError("Must provide a non-empty list of channel keys.")

    shuffle = (shuffle_buffer_size > 0)
    deterministic = (not shuffle)

    if tfr_channel_rewrite_map:
        channel_rewrite_map = {
            tfr_channel_keys.index(k): v
            for k, v in tfr_channel_rewrite_map.items()
        }
    else:
        channel_rewrite_map = {}

    def _parse_example(serialized):
        """Parses a serialized tf.train.Example into an (image, label) tuple."""
        # pylint: disable=no-value-for-parameter

        all_keys = tfr_channel_keys + [tfr_label_key]
        example = tf.io.parse_example(
            serialized,
            {key: tf.io.FixedLenFeature([], tf.string) for key in all_keys})

        def _build_image_from_channels(*channels_data):
            img = []
            for idx, data in enumerate(channels_data):
                with MemoryFile(data) as memfile:
                    with memfile.open() as f:
                        channel = f.read(1)
                        if idx in channel_rewrite_map:
                            v_from, v_to = channel_rewrite_map[idx]
                            channel[channel == v_from] = v_to
                        img.append(channel)
            img = np.stack(img, axis=-1)
            return img

        img = tf.numpy_function(
            func=_build_image_from_channels,
            inp=[example[key] for key in tfr_channel_keys],  # type: ignore
            Tout=tf.float32)

        def _read_label(data):
            with MemoryFile(data) as memfile:
                with memfile.open() as f:
                    label = f.read(1)
            return np.expand_dims(label, axis=-1)

        label = tf.numpy_function(func=_read_label,
                                  inp=[example[tfr_label_key]],
                                  Tout=tf.uint8)

        return img, label

    ds = tf.data.TFRecordDataset.list_files(str(file_pattern), shuffle=shuffle)
    ds = ds.interleave(tf.data.TFRecordDataset,
                       num_parallel_calls=tf.data.AUTOTUNE,
                       deterministic=deterministic)
    if repeat:
        ds = ds.repeat()
    if shuffle:
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    ds = ds.map(_parse_example,
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=deterministic)

    if transforms:
        ds = augment_image_dataset(ds,
                                   transforms=transforms,
                                   augment_labels=True)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = \
        tf.data.experimental.AutoShardPolicy.DATA
    ds = ds.with_options(options)
    ds = ds.batch(batch_size=batch_size,
                  drop_remainder=True,
                  num_parallel_calls=tf.data.AUTOTUNE,
                  deterministic=deterministic)

    ds = ds.prefetch(prefetch_buffer_size)
    return ds
