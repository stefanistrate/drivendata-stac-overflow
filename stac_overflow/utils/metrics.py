"""Custom metrics."""

import tensorflow as tf


class MaskedIOU(tf.keras.metrics.Metric):
    """Masked mean intersection-over-union, defined at pixel level."""

    def __init__(self, name="masked_iou", **kwargs):
        super().__init__(name=name, **kwargs)
        self.intersection = self.add_weight(name="intersection",
                                            initializer="zeros")
        self.union = self.add_weight(name="union", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        del sample_weight  # Unused.

        mask = tf.not_equal(y_true, 255)
        y_true = tf.cast(y_true, tf.bool)
        y_true = tf.where(mask, y_true, False)
        y_pred = tf.where(y_pred > 0.5, True, False)
        y_pred = tf.where(mask, y_pred, False)

        intersections = tf.math.logical_and(y_true, y_pred)
        intersections = tf.cast(intersections, tf.float32)
        unions = tf.math.logical_or(y_true, y_pred)
        unions = tf.cast(unions, tf.float32)

        self.intersection.assign_add(tf.reduce_sum(intersections))
        self.union.assign_add(tf.reduce_sum(unions))

    def reset_state(self):
        self.intersection.assign(0)
        self.union.assign(0)

    def result(self):
        return tf.math.divide_no_nan(self.intersection, self.union)
