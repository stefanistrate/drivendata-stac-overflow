# [STAC Overflow: map floodwater from radar imagery](https://www.drivendata.org/competitions/81/detect-flood-water/page/385/)

[![Python build](https://github.com/stefanistrate/drivendata-stac-overflow/actions/workflows/python-build.yml/badge.svg)](https://github.com/stefanistrate/drivendata-stac-overflow/actions/workflows/python-build.yml)

Code for the STAC Overflow competition on DrivenData.

## Full production installation

```bash
pip install -e .
```

## Full development installation

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
  --train_steps_per_epoch=10 \
  --num_epochs=5 \
  --data_augmentation \
  --wandb_project=stac_overflow \
  --wandb_group=local \
  --wandb_mode=online

done
```

## Grid.ai GPU training with k-fold cross-validation

Check machine specs and hourly rates
[here](https://docs.grid.ai/platform/billing-rates#individual-tier-hourly-rates).

TODO(stefanistrate): Update boolean flags when
[this Grid.ai bug](https://github.com/gridai/gridai/issues/134) is fixed.

### On a single GPU

```bash
VALIDATION=("awc,ayt,hxu,tnp" "ayt,hxu,pxs,qus" "coz,hxu,jja,tht" "coz,kuo,tht,wvy" "hbe,hxu,kuo,qus")

for VAL in $VALIDATION ; do

grid run \
  --instance_type=g4dn.xlarge \
  --cpus=3 \
  --gpus=1 \
  --dockerfile=stac_overflow/Dockerfile \
  stac_overflow/train_segmentation_model.py \
  --root_tfrecords="grid:stac_overflow_tfrecords:10" \
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
  --wandb_project=stac_overflow \
  --wandb_group=unet_pc_rewrite_255s \
  --wandb_mode=online

done
```
