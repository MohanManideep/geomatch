# GeoMatch — photo geolocation by visual retrieval

Single-model image geolocation for a 12-country European dataset (BY, DE, ES,
FI, FR, GB, IS, IT, NO, PL, SE, TR). Given a photo, the model predicts a
latitude/longitude by matching it against a bank of training images and reading
the coordinate of the best match.

## Model

`GeoCPRegNetRetrieval` (`src/retrieval_model.py`):

- A single RegNetY-400MF trunk trained **from scratch** (`weights=None`), shared
  across three views of each image: the full frame at 512 px plus aligned
  left/right crops at 256 px.
- Multi-scale GeM pooling over stages 2–4 into a 384-D global descriptor, plus
  48 local tokens (3 views x a 4x4 grid, 64-D each) for a late-interaction
  (MaxSim) rerank.
- **4,869,911 trainable parameters**.

Decoding: cosine top-200 shortlist on the global descriptor, rerank by
`0.2 * global + 0.8 * local`, take the top-1 candidate's coordinate.

![Model pipeline](images/geomatch.png)

## Training

Two stages, run per fold (train on 4 folds, evaluate on the held-out
one, fixed epoch count, no checkpoint selection):

1. **Self-supervised backbone pretraining** (`src/ssl_pretrain_backbone.py`) —
   BYOL (negative-free, momentum target + predictor), labels unused.
2. **Country-aware retrieval fine-tuning**
   (`src/country_aware_retrieval_finetune.py`, config `configs/final_recipe.json`)
   — spatial listwise + local listwise + triplet + local-verification losses,
   geographic positives within 35 km, same-country hard negatives re-mined each
   epoch, anchor oversampling for the weakest countries (DE, FR, PL, IT, ES, SE,
   GB), EMA teacher, 40 epochs.

Training time (single RTX 4000 Ada, 20 GB): BYOL backbone ≈ 2 h 40 m per fold;
retrieval finetune ≈ 70 min (40 epochs). The submitted all-data model reuses an
existing backbone, so end to end it is ≈ 70 min plus ≈ 5 min to write
`predictions.csv`.

## Results

Pooled 5-fold out-of-fold median haversine error: **55.6 km** (best of three
seeds; 49.1% of images within 50 km). The submitted model retrains the same
recipe on all 11,758 images.

Self-supervised pretraining plus the country-aware retrieval finetune cut the
per-country error across the board versus a plain from-scratch retrieval
baseline — except in Germany and France, which stay far behind every other
country:

![Median error by country](images/figures/per_country_median.png)

The bottleneck is ranking, not recall: a within-50 km training image is in the
retrieval top-200 for ~80% of queries, but the decoder selects it only ~49% of
the time, and the gap is worst for DE/FR.

![Oracle vs decoded](images/figures/oracle_gap.png)

![Where the misses are](images/figures/error_map.png)

## Layout

```
configs/            final_recipe.json (training recipe), data.json (view + augmentation spec)
src/                model, data pipeline, objectives, and the two training entry points
splits/             official_folds_seed42.csv
artifacts/          per-fold normalization stats and geo-cell assignments read by the pipeline
images/             figure-generation script and the figures used in the write-up
EXPERIMENT_LOG.md   narrative record of what was tried and why
EXPERIMENTS_TABLE.md the same history as a compact experiment/result/issue table
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Requires a CUDA GPU with bfloat16 support for training. Image data is expected at
`/var/tmp/luli38se-geomatch/data/geo_dataset/train`; override with `--images
<path>` on either training script.

## Run

### Cross-validation (produces the reported 5-fold OOF numbers)

```bash
# 1. SSL backbone, one per fold
for f in 0 1 2 3 4; do
  python src/ssl_pretrain_backbone.py --fold $f
done

# 2. retrieval fine-tune across all folds
python src/country_aware_retrieval_finetune.py \
    --config configs/final_recipe.json --folds 0 1 2 3 4
```

### Submitted model (all 11,758 images, no held-out fold)

```bash
# retrain the recipe on every labelled image (resume-safe)
python src/full_data_finetune.py --resume

# generate predictions.csv for the public holdout set
python src/predict_holdout.py
```

`full_data_finetune.py` initialises the trunk from one of the BYOL backbones
above and finetunes for 40 epochs on all labelled data; `predict_holdout.py`
runs the top-200 shortlist + late-interaction rerank and writes
`predictions.csv` (`filename,pred_lat,pred_lng`, 2400 rows).

```bash
# regenerate report figures from saved predictions
python images/make_figures.py
```

## Formatting

```bash
pip install black isort
isort src images && black src images
```

Configuration is in `pyproject.toml` (black + isort, 88-column).
