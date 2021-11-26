"""Tests for datasets utilities."""

import albumentations
import numpy as np
import tensorflow as tf

from stac_overflow.utils.datasets import augment_image_dataset


class TestAugmentImageDataset:
    """Test `augment_image_dataset()`."""

    def _build_dataset(self):
        return tf.data.Dataset.from_tensor_slices((
            # images
            [
                [
                    [[-1], [-2], [-3], [-4]],
                    [[-5], [-6], [-7], [-8]],
                    [[-9], [10], [11], [12]],
                ],
                [
                    [[101], [102], [103], [104]],
                    [[105], [106], [107], [108]],
                    [[109], [110], [111], [112]],
                ],
            ],
            # labels
            [
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                ],
                [
                    [0, 0, 1, 0],
                    [0, 1, 0, 0],
                    [1, 0, 0, 0],
                ],
            ],
        ))

    def test_without_augment_labels(self):
        ds = self._build_dataset()
        transforms = albumentations.Compose([albumentations.Transpose(p=1.0)])
        new_ds = augment_image_dataset(ds, transforms)
        new_ds = list(new_ds.as_numpy_iterator())
        assert len(new_ds) == 2  # list
        assert len(new_ds[0]) == 2  # tuple
        np.testing.assert_array_equal(new_ds[0][0], [
            [[-1], [-5], [-9]],
            [[-2], [-6], [10]],
            [[-3], [-7], [11]],
            [[-4], [-8], [12]],
        ])
        np.testing.assert_array_equal(new_ds[0][1], [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ])
        assert len(new_ds[1]) == 2  # tuple
        np.testing.assert_array_equal(new_ds[1][0], [
            [[101], [105], [109]],
            [[102], [106], [110]],
            [[103], [107], [111]],
            [[104], [108], [112]],
        ])
        np.testing.assert_array_equal(new_ds[1][1], [
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
        ])

    def test_with_augment_labels(self):
        ds = self._build_dataset()
        transforms = albumentations.Compose([albumentations.Transpose(p=1.0)])
        new_ds = augment_image_dataset(ds, transforms, augment_labels=True)
        new_ds = list(new_ds.as_numpy_iterator())
        assert len(new_ds) == 2  # list
        assert len(new_ds[0]) == 2  # tuple
        np.testing.assert_array_equal(new_ds[0][0], [
            [[-1], [-5], [-9]],
            [[-2], [-6], [10]],
            [[-3], [-7], [11]],
            [[-4], [-8], [12]],
        ])
        np.testing.assert_array_equal(new_ds[0][1], [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ])
        assert len(new_ds[1]) == 2  # tuple
        np.testing.assert_array_equal(new_ds[1][0], [
            [[101], [105], [109]],
            [[102], [106], [110]],
            [[103], [107], [111]],
            [[104], [108], [112]],
        ])
        np.testing.assert_array_equal(new_ds[1][1], [
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 0],
        ])
