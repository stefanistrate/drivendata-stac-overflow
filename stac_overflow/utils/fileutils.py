"""File utilities."""

import tensorflow as tf

from absl import logging
from urllib.request import urlretrieve


def load_local_or_remote_h5_model(
        filename_or_url: str,
        compile: bool = False,  # pylint: disable=redefined-builtin
        custom_objects=None) -> tf.keras.Model:
    if (filename_or_url.startswith("http://") or
            filename_or_url.startswith("https://")):
        logging.info("Downloading model from URL: %s", filename_or_url)
        urlretrieve(filename_or_url, "init_model.h5")
        logging.info("Finished downloading model to `init_model.h5`.")
        filename_or_url = "init_model.h5"

    return tf.keras.models.load_model(filename_or_url,
                                      custom_objects=custom_objects,
                                      compile=compile)
