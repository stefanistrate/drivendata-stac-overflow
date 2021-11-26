"""Custom losses."""

import tensorflow as tf


class MaskedBCEWithDiceLoss(tf.keras.losses.Loss):
    """Masked loss: 50% binary cross entropy + 50% Dice."""

    def __init__(self, name="masked_bce_dice", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        mask = tf.not_equal(y_true, 255)
        y_true = tf.cast(y_true, tf.float32)
        y_true = tf.where(mask, y_true, 0.0)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.where(mask, y_pred, 0.0)

        bce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        dice_loss = (1 - tf.math.divide_no_nan(
            2 * tf.reduce_sum(y_true * y_pred), tf.reduce_sum(y_true + y_pred)))

        return (bce_loss + dice_loss) / 2


class MaskedDiceLoss(tf.keras.losses.Loss):
    """Masked Dice loss."""

    def __init__(self, name="masked_dice", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        mask = tf.not_equal(y_true, 255)
        y_true = tf.cast(y_true, tf.float32)
        y_true = tf.where(mask, y_true, 0.0)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.where(mask, y_pred, 0.0)

        dice_loss = (1 - tf.math.divide_no_nan(
            2 * tf.reduce_sum(y_true * y_pred), tf.reduce_sum(y_true + y_pred)))
        return dice_loss
