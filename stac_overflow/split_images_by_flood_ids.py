"""Split images by flood ids.

The outputs will be in TFRecord format.

Note: The implementation uses Apache Beam for defining the processing pipeline.
Apache Beam does pickling, which sometimes results in NameErrors on imports.
The recommended workaround is to import those modules locally:
https://cloud.google.com/dataflow/docs/resources/faq#how_do_i_handle_nameerrors
"""

# Do not call `super()` in `__init__()` due to this pickling bug:
# https://github.com/uqfoundation/dill/issues/300
# pylint: disable=super-init-not-called

from __future__ import annotations

import apache_beam as beam
import pandas as pd

from absl import app
from absl import flags
from absl import logging
from apache_beam.options.pipeline_options import SetupOptions
from pathlib import Path

from stac_overflow.utils.logging import print_pipeline_result

FLAGS = flags.FLAGS

flags.DEFINE_string("metadata_csv", None, "Metadata CSV.", required=True)
flags.DEFINE_list("validation_flood_ids", "_",
                  ("List of flood ids to include in the validation set. "
                   "All the other flood ids go to the training set. "
                   "If `_`, validation set will be empty."))
flags.DEFINE_string("features_dir",
                    None,
                    "Directory of input features.",
                    required=True)
flags.DEFINE_string("labels_dir",
                    None,
                    "Directory of target labels.",
                    required=True)
flags.DEFINE_string("destination_dir",
                    None,
                    "Destination directory for data splits.",
                    required=True)
flags.DEFINE_integer("num_shards", 10, "Number of shards for data splits.")


def load_chips(metadata_csv: Path) -> list[dict]:
    """Loads chips as a list of dicts."""

    metadata = pd.read_csv(metadata_csv)
    chips = []
    for chip_id, group in metadata.groupby("chip_id"):
        flood_id = group["flood_id"].values[0]
        vv_path = group[group.polarization ==
                        "vv"]["image_id"].values[0] + ".tif"
        vh_path = group[group.polarization ==
                        "vh"]["image_id"].values[0] + ".tif"
        nasadem_path = chip_id + "_nasadem.tif"
        jrc_change_path = chip_id + "_jrc-gsw-change.tif"
        jrc_extent_path = chip_id + "_jrc-gsw-extent.tif"
        jrc_occurrence_path = chip_id + "_jrc-gsw-occurrence.tif"
        jrc_recurrence_path = chip_id + "_jrc-gsw-recurrence.tif"
        jrc_seasonality_path = chip_id + "_jrc-gsw-seasonality.tif"
        jrc_transitions_path = chip_id + "_jrc-gsw-transitions.tif"
        label_path = chip_id + ".tif"
        chips.append({
            "chip_id": chip_id,
            "flood_id": flood_id,
            "vv_path": vv_path,
            "vh_path": vh_path,
            "nasadem_path": nasadem_path,
            "jrc_change_path": jrc_change_path,
            "jrc_extent_path": jrc_extent_path,
            "jrc_occurrence_path": jrc_occurrence_path,
            "jrc_recurrence_path": jrc_recurrence_path,
            "jrc_seasonality_path": jrc_seasonality_path,
            "jrc_transitions_path": jrc_transitions_path,
            "label_path": label_path,
        })
    return chips


class SplitByFloodId(beam.DoFn):
    """DoFn that splits chips dataset by flood id."""

    def process(self, element, validation_flood_ids):

        # Workaround for NameErrors.
        # pylint: disable=import-outside-toplevel
        # pylint: disable=reimported
        # pylint: disable=redefined-outer-name
        import apache_beam as beam
        # pylint: enable=redefined-outer-name
        # pylint: enable=reimported
        # pylint: enable=import-outside-toplevel

        if element["flood_id"] in validation_flood_ids:
            yield beam.pvalue.TaggedOutput("validation", element)
        else:
            yield beam.pvalue.TaggedOutput("train", element)


class BuildTFRecord(beam.DoFn):
    """DoFn that builds a TFRecord for a given chip."""

    def __init__(self):
        self.num_outputs = beam.metrics.Metrics.counter("main", "num_outputs")
        self.num_chip_errors = beam.metrics.Metrics.counter(
            "main", "num_chip_errors")

    def process(self, element, features_dir: Path, labels_dir: Path):

        # Workaround for NameErrors.
        # pylint: disable=import-outside-toplevel
        # pylint: disable=reimported
        # pylint: disable=redefined-outer-name
        # yapf: disable
        import tensorflow as tf
        from absl import logging
        # yapf: enable
        # pylint: enable=redefined-outer-name
        # pylint: enable=reimported
        # pylint: enable=import-outside-toplevel

        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(
                value=[value]))

        try:
            vv = (features_dir / element["vv_path"]).read_bytes()
            vh = (features_dir / element["vh_path"]).read_bytes()
            nasadem = (features_dir / element["nasadem_path"]).read_bytes()
            jrc_change = (features_dir /
                          element["jrc_change_path"]).read_bytes()
            jrc_extent = (features_dir /
                          element["jrc_extent_path"]).read_bytes()
            jrc_occurrence = (features_dir /
                              element["jrc_occurrence_path"]).read_bytes()
            jrc_recurrence = (features_dir /
                              element["jrc_recurrence_path"]).read_bytes()
            jrc_seasonality = (features_dir /
                               element["jrc_seasonality_path"]).read_bytes()
            jrc_transitions = (features_dir /
                               element["jrc_transitions_path"]).read_bytes()
            label = (labels_dir / element["label_path"]).read_bytes()
        except OSError:
            self.num_chip_errors.inc()
            logging.error("Error processing chip: %s", element["chip_id"])
            return

        feature = {
            "vv": _bytes_feature(vv),
            "vh": _bytes_feature(vh),
            "nasadem": _bytes_feature(nasadem),
            "jrc_change": _bytes_feature(jrc_change),
            "jrc_extent": _bytes_feature(jrc_extent),
            "jrc_occurrence": _bytes_feature(jrc_occurrence),
            "jrc_recurrence": _bytes_feature(jrc_recurrence),
            "jrc_seasonality": _bytes_feature(jrc_seasonality),
            "jrc_transitions": _bytes_feature(jrc_transitions),
            "label": _bytes_feature(label),
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        self.num_outputs.inc()
        logging.log_every_n(logging.INFO, "Built another %d TFRecords.", 1000,
                            1000)
        yield example.SerializeToString()


def build_tfrecords(pipeline: beam.Pipeline, features_dir: Path,
                    labels_dir: Path, destination_dir: Path, split: str,
                    num_shards: int):
    """Builds and writes TFRecords to a destination directory."""

    output_path = destination_dir / split

    tfrecords = (
        pipeline
        | f"TFRecords '{split}': build records"
        >> beam.ParDo(BuildTFRecord(), features_dir, labels_dir)
        | f"TFRecords '{split}': shuffle records"
        >> beam.Reshuffle()
    )  # yapf: disable
    _ = (
        tfrecords
        | f"TFRecords '{split}': write records"
        >> beam.io.tfrecordio.WriteToTFRecord(
            str(output_path),
            file_name_suffix=".tfrecords",
            num_shards=num_shards)
    )  # yapf: disable



def create_pipeline(beam_args: list[str], chips_list: list[dict],
                    validation_flood_ids: list[str], features_dir: Path,
                    labels_dir: Path, destination_dir: Path,
                    num_shards: int) -> beam.Pipeline:

    pipeline = beam.Pipeline(options=SetupOptions(beam_args))

    chips = (
        pipeline
        | "Chips: load" >> beam.Create(chips_list)
        | "Chips: split by flood id"
        >> beam
        .ParDo(SplitByFloodId(), validation_flood_ids=validation_flood_ids)
        .with_outputs("train", "validation")
    )  # yapf: disable

    build_tfrecords(chips.train, features_dir, labels_dir, destination_dir,
                    "train", num_shards)
    build_tfrecords(chips.validation, features_dir, labels_dir, destination_dir,
                    "validation", num_shards)

    return pipeline


def main(argv):
    log_dir = Path(FLAGS.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.get_absl_handler().use_absl_log_file(log_dir=log_dir)

    chips = load_chips(Path(FLAGS.metadata_csv))
    if FLAGS.validation_flood_ids == "_":
        validation_flood_ids = []
        destination_dir = Path(FLAGS.destination_dir, "_")
    else:
        validation_flood_ids = FLAGS.validation_flood_ids
        destination_dir = Path(FLAGS.destination_dir,
                               ",".join(FLAGS.validation_flood_ids))

    pipeline = create_pipeline(argv[1:], chips, validation_flood_ids,
                               Path(FLAGS.features_dir), Path(FLAGS.labels_dir),
                               destination_dir, FLAGS.num_shards)
    pipeline_result = pipeline.run()
    logging.info("Successful run!")
    print_pipeline_result(pipeline_result, print_fn=logging.info)


if __name__ == "__main__":
    app.run(main)
