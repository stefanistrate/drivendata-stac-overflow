"""Binary image segmentation model."""

from __future__ import annotations

import albumentations
import cv2
import os
import sys
import segmentation_models as sm
import tensorflow as tf
import tensorflow_addons as tfa
import wandb

from absl import app
from absl import flags
from absl import logging
from datetime import datetime
from pathlib import Path
from pathlib import PurePath

from stac_overflow.submission.inputs import parse_channels_and_rewrites
from stac_overflow.utils.datasets import tfrecords_as_geospatial_dataset
from stac_overflow.utils.fileutils import load_local_or_remote_h5_model
from stac_overflow.utils.logging import print_model_summary
from stac_overflow.utils.logging import sg_pl
from stac_overflow.utils.losses import MaskedDiceLoss
from stac_overflow.utils.metrics import MaskedIOU

# Inputs and outputs.
flags.DEFINE_string(
    "root_tfrecords", None,
    ("Root directory for train and validation TFRecords. "
     "If specified, `FLAGS.train_tfrecords` and `FLAGS.validation_tfrecords` "
     "will be relative file patterns."))
flags.DEFINE_string("train_tfrecords",
                    None,
                    "File pattern for train TFRecords.",
                    required=True)
flags.DEFINE_string("validation_tfrecords",
                    None,
                    "File pattern for validation TFRecords.",
                    required=True)
flags.DEFINE_list(
    "tfrecords_geo_channel_keys",
    None, "List of keys representing the geospatial data in the tfrecords. "
    "Any key can also be followed by a rewriting rule "
    "(e.g. `key1:from1:to1,key2:from2:to2`), which will specify which invalid "
    "integer value has to be rewritten.",
    required=True)
flags.DEFINE_string("tfrecords_geo_label_key", "label",
                    "Key representing the label in the tfrecords.")
flags.DEFINE_integer("img_height",
                     None,
                     "Height of input images.",
                     required=True)
flags.DEFINE_integer("img_width", None, "Width of input images.", required=True)
flags.DEFINE_string("models_dir", None, "Models directory.", required=True)

# Model architecture.
flags.DEFINE_enum("network_type",
                  None, ["unet", "linknet", "fpn"],
                  "Network type.",
                  required=True)
flags.DEFINE_enum("backbone",
                  None,
                  sm.get_available_backbone_names(),
                  "Backbone network.",
                  required=True)
flags.DEFINE_string("init_model", None, "SavedModel to initialize from.")

# Training configuration.
flags.DEFINE_integer("num_replicas", 1, "Number of replicas in use.")
flags.DEFINE_integer(
    "batch_size_per_replica", 8,
    ("Batch size for a single replica. "
     "Global batch size = num_replicas * batch_size_per_replica."))
flags.DEFINE_integer("train_steps_per_epoch", 10,
                     "Number of train steps per epoch.")
flags.DEFINE_integer("num_epochs", 1, "Number of training epochs.")
flags.DEFINE_float("learning_rate", 1e-3, "Learning rate for optimizer.")
flags.DEFINE_integer(
    "shuffle_buffer_size", 1024,
    "Buffer size for dataset shuffling. If `0`, disable shuffling.")
flags.DEFINE_integer("prefetch_buffer_size", tf.data.AUTOTUNE,
                     "Buffer size for dataset prefetching.")
flags.DEFINE_bool("data_augmentation", False,
                  "Whether to augment data at training time.")

# Callbacks configuration.
flags.DEFINE_bool("early_stopping", False,
                  "Whether to use early stopping at training time.")
flags.DEFINE_bool("progress_bar", True,
                  "Whether to show the progress bar during training.")
flags.DEFINE_integer("time_limit", 8 * 3600,
                     "Time limit (in seconds) for each `model.fit()` call.")

# Logging configuration.
flags.DEFINE_bool(
    "redirect_logs", False,
    "Whether to redirect stdout, stderr and absl.logging to a log file.")
flags.DEFINE_string("wandb_api_key", None, "API key for W&B logging.")
flags.DEFINE_string("wandb_project", None, "Project name for W&B logging.")
flags.DEFINE_string("wandb_group", None, "Group name for W&B logging.")
flags.DEFINE_enum("wandb_mode", "disabled", ["online", "offline", "disabled"],
                  "Running mode for W&B logging.")

FLAGS = flags.FLAGS


def prepare_augmentations():
    return albumentations.Compose([
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.Transpose(p=0.5),
        albumentations.Rotate(limit=[-180, 180],
                              interpolation=cv2.INTER_NEAREST,
                              p=0.5),
    ])


def build_model(network_type: str = None,
                backbone: str = None,
                channels: int = None) -> tf.keras.Model:
    sm.set_framework("tf.keras")
    if network_type == "unet":
        base_model = sm.Unet(backbone_name=backbone,
                             input_shape=[None, None, channels],
                             classes=1,
                             activation="sigmoid",
                             encoder_weights=None)
    elif network_type == "linknet":
        base_model = sm.Linknet(backbone_name=backbone,
                                input_shape=[None, None, channels],
                                classes=1,
                                activation="sigmoid",
                                encoder_weights=None)
    elif network_type == "fpn":
        base_model = sm.FPN(backbone_name=backbone,
                            input_shape=[None, None, channels],
                            classes=1,
                            activation="sigmoid",
                            encoder_weights=None)
    else:
        raise ValueError(f"Invalid network type: {network_type}.")

    input_img = tf.keras.Input(shape=[None, None, channels])
    output_segmentation = base_model(input_img)
    model = tf.keras.Model(input_img, output_segmentation, name=base_model.name)
    return model


def compile_model(model: tf.keras.Model, learning_rate: float):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss=MaskedDiceLoss(),
                  metrics=[MaskedIOU()])


def load_model(init_model: str) -> tf.keras.Model:
    return load_local_or_remote_h5_model(init_model,
                                         compile=True,
                                         custom_objects={
                                             "MaskedDiceLoss": MaskedDiceLoss(),
                                             "MaskedIOU": MaskedIOU(),
                                         })


def main(argv):
    del argv  # Unused.

    # Artifacts configuration.
    timestamp = f"{datetime.now():%Y-%m-%d-%H%M%S}"
    model_dir = Path(FLAGS.models_dir, FLAGS.backbone)
    log_dir = model_dir / f"logs-{timestamp}"
    if FLAGS.redirect_logs:
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.get_absl_handler().use_absl_log_file(log_dir=log_dir)
        sys.stdout = logging.get_absl_handler().python_handler.stream
        sys.stderr = logging.get_absl_handler().python_handler.stream

    # Weights & Biases configuration.
    if FLAGS.wandb_api_key:
        os.environ["WANDB_API_KEY"] = FLAGS.wandb_api_key
    wandb.tensorboard.patch(root_logdir=str(log_dir))
    wandb.init(project=FLAGS.wandb_project,
               name=os.environ.get("GRID_EXPERIMENT_NAME"),
               group=FLAGS.wandb_group,
               config=FLAGS,
               tags=filter(None, [
                   "tensorflow",
                   FLAGS.network_type,
                   FLAGS.backbone,
                   f"img_{FLAGS.img_height}x{FLAGS.img_width}",
                   os.environ.get("GRID_INSTANCE_TYPE"),
               ]),
               mode=FLAGS.wandb_mode)

    # Choose running strategy.
    strategy = tf.distribute.MultiWorkerMirroredStrategy()

    # Load data.
    batch_size = FLAGS.num_replicas * FLAGS.batch_size_per_replica
    train_tfrecords = (PurePath(FLAGS.root_tfrecords, FLAGS.train_tfrecords)
                       if FLAGS.root_tfrecords else PurePath(
                           FLAGS.train_tfrecords))
    if FLAGS.data_augmentation:
        augmentations = prepare_augmentations()
    else:
        augmentations = None
    channel_keys, channel_rewrite_map = parse_channels_and_rewrites(
        FLAGS.tfrecords_geo_channel_keys)
    ds_train = tfrecords_as_geospatial_dataset(
        file_pattern=train_tfrecords,
        batch_size=batch_size,
        repeat=True,
        shuffle_buffer_size=FLAGS.shuffle_buffer_size,
        prefetch_buffer_size=FLAGS.prefetch_buffer_size,
        transforms=augmentations,
        tfr_channel_keys=channel_keys,
        tfr_channel_rewrite_map=channel_rewrite_map,
        tfr_label_key=FLAGS.tfrecords_geo_label_key)
    validation_tfrecords = (
        PurePath(FLAGS.root_tfrecords, FLAGS.validation_tfrecords)
        if FLAGS.root_tfrecords else PurePath(FLAGS.validation_tfrecords))
    ds_validation = tfrecords_as_geospatial_dataset(
        file_pattern=validation_tfrecords,
        batch_size=batch_size,
        repeat=False,
        shuffle_buffer_size=0,
        prefetch_buffer_size=FLAGS.prefetch_buffer_size,
        tfr_channel_keys=channel_keys,
        tfr_channel_rewrite_map=channel_rewrite_map,
        tfr_label_key=FLAGS.tfrecords_geo_label_key)

    # Prepare training callbacks.
    save_options = tf.saved_model.SaveOptions(
        experimental_custom_gradients=True)
    best_model_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_dir / f"best-epoch{{epoch}}-{timestamp}.h5",
        save_best_only=True,
        options=save_options)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10)
    lr_cb = tf.keras.callbacks.ReduceLROnPlateau(factor=0.1,
                                                 patience=3,
                                                 min_lr=1e-5)
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    tqdm_cb = tfa.callbacks.TQDMProgressBar()
    time_stopping_cb = tfa.callbacks.TimeStopping(seconds=FLAGS.time_limit,
                                                  verbose=1)
    callbacks = list(
        filter(None, [
            best_model_cb,
            early_stopping_cb if FLAGS.early_stopping else None,
            lr_cb,
            tensorboard_cb,
            tqdm_cb if FLAGS.progress_bar else None,
            time_stopping_cb,
        ]))

    # Initialize the model from a given SavedModel or build from scratch.
    if FLAGS.init_model:
        with strategy.scope():
            model = load_model(FLAGS.init_model)
            print_model_summary(model, print_fn=logging.info)
    else:
        with strategy.scope():
            model = build_model(network_type=FLAGS.network_type,
                                backbone=FLAGS.backbone,
                                channels=len(FLAGS.tfrecords_geo_channel_keys))
            compile_model(model, FLAGS.learning_rate)
            print_model_summary(model, print_fn=logging.info)

    # Train the model.
    logging.info("Training the model for %s...",
                 sg_pl(FLAGS.num_epochs, "epoch", "epochs"))
    model.fit(x=ds_train,
              validation_data=ds_validation,
              epochs=FLAGS.num_epochs,
              steps_per_epoch=FLAGS.train_steps_per_epoch,
              use_multiprocessing=True,
              callbacks=callbacks,
              verbose=0 if FLAGS.progress_bar else 2)

    # Save the model.
    logging.info("Saving the final model...")
    model.save(model_dir / f"model-{timestamp}.h5", options=save_options)


if __name__ == "__main__":
    app.run(main)
