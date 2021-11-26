# STAC Overflow: Map Floodwater from Radar Imagery

[![Python build](https://github.com/stefanistrate/drivendata-stac-overflow/actions/workflows/python-build.yml/badge.svg)](https://github.com/stefanistrate/drivendata-stac-overflow/actions/workflows/python-build.yml)

Code for the
[STAC Overflow competition](https://www.drivendata.org/competitions/81/detect-flood-water/page/385/)
on DrivenData. Originally
[ranked 12th](https://www.drivendata.org/competitions/81/detect-flood-water/leaderboard/)
with a single-model solution, but then developed even further by implementing an
ensemble model.

## Local installation for development

```bash
pip install -e .[dev]

python -m pretty_errors -s -p
```

## Data splitting for train/validation

```bash
DATASTORE="$HOME/datastore/stac-overflow"

VALIDATION=("_" "awc,ayt,hxu,tnp" "ayt,hxu,pxs,qus" "coz,hxu,jja,tht" "coz,kuo,tht,wvy" "hbe,hxu,kuo,qus")

for VAL in $VALIDATION ; do

python stac_overflow/split_images_by_flood_ids.py \
  --metadata_csv="$DATASTORE/flood-training-metadata.csv" \
  --validation_flood_ids="$VAL" \
  --features_dir="$DATASTORE/train_features" \
  --labels_dir="$DATASTORE/train_labels" \
  --destination_dir="$DATASTORE/tfrecords" \
  --log_dir="$DATASTORE/tfrecords_logs" \
  --alsologtostderr \
  -- \
  --runner=DirectRunner \
  --direct_running_mode=multi_processing \
  --direct_num_workers=0

done
```

## Local training with k-fold cross-validation

```bash
DATASTORE="$HOME/datastore/stac-overflow"

VALIDATION=("awc,ayt,hxu,tnp" "ayt,hxu,pxs,qus" "coz,hxu,jja,tht" "coz,kuo,tht,wvy" "hbe,hxu,kuo,qus")

for VAL in $VALIDATION ; do

python stac_overflow/train_segmentation_model.py \
  --train_tfrecords="$DATASTORE/tfrecords/$VAL/train-*.tfrecords" \
  --validation_tfrecords="$DATASTORE/tfrecords/$VAL/validation-*.tfrecords" \
  --tfrecords_geo_channel_keys="vv,vh,nasadem,jrc_extent:255:0,jrc_occurrence:255:0,jrc_recurrence:255:0,jrc_seasonality:255:0,jrc_transitions:255:0" \
  --img_height=512 \
  --img_width=512 \
  --models_dir="$DATASTORE/models" \
  --network_type="unet" \
  --backbone="seresnet152" \
  --num_replicas=1 \
  --batch_size_per_replica=4 \
  --train_steps_per_epoch=5 \
  --num_epochs=5 \
  --data_augmentation \
  --wandb_project=stac-overflow \
  --wandb_group=local \
  --wandb_mode=online

done
```

## [Grid.ai](https://www.grid.ai/) training with k-fold cross-validation

Check machine specs and hourly rates
[here](https://docs.grid.ai/platform/billing-rates#individual-tier-hourly-rates).

TODO(stefanistrate): Remove dummy `true` values from the boolean flags below
after [this Grid.ai bug](https://github.com/gridai/gridai/issues/134) is fixed.

```bash
VALIDATION=("awc,ayt,hxu,tnp" "ayt,hxu,pxs,qus" "coz,hxu,jja,tht" "coz,kuo,tht,wvy" "hbe,hxu,kuo,qus")

for VAL in $VALIDATION ; do

grid run \
  --instance_type=g4dn.xlarge \
  --cpus=3 \
  --gpus=1 \
  --dockerfile=Dockerfile \
  stac_overflow/train_segmentation_model.py \
  --root_tfrecords="grid:stac-overflow:1" \
  --train_tfrecords="$VAL/train-*.tfrecords" \
  --validation_tfrecords="$VAL/validation-*.tfrecords" \
  --tfrecords_geo_channel_keys="vv,vh,nasadem,jrc_extent:255:0,jrc_occurrence:255:0,jrc_recurrence:255:0,jrc_seasonality:255:0,jrc_transitions:255:0" \
  --img_height=512 \
  --img_width=512 \
  --models_dir="models" \
  --network_type="unet" \
  --backbone="seresnet152" \
  --num_replicas=1 \
  --batch_size_per_replica=4 \
  --train_steps_per_epoch=100 \
  --num_epochs=50 \
  --data_augmentation=true \
  --noearly_stopping=true \
  --noprogress_bar=true \
  --redirect_logs=true \
  --wandb_api_key="$WANDB_API_KEY" \
  --wandb_project=stac-overflow \
  --wandb_group=unet_pc_rewrite_255s \
  --wandb_mode=online

done
```
