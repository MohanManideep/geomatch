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
- **4,869,911 trainable parameters** (under the 5,000,000 limit; checked at
  construction and by `python src/spatial_retrieval_finetune.py --self-test`).

Decoding: cosine top-200 shortlist on the global descriptor, rerank by
`0.2 * global + 0.8 * local`, take the top-1 candidate's coordinate.

## Training

Two stages, run per official fold (train on 4 folds, evaluate on the held-out
one, fixed epoch count, no checkpoint selection):

1. **Self-supervised backbone pretraining** (`src/ssl_pretrain_backbone.py`) —
   BYOL (negative-free, momentum target + predictor), 160 epochs, labels unused.
2. **Country-aware retrieval fine-tuning**
   (`src/country_aware_retrieval_finetune.py`, config `configs/final_recipe.json`)
   — spatial listwise + local listwise + triplet + local-verification losses,
   geographic positives within 35 km, same-country hard negatives re-mined each
   epoch, anchor oversampling for the weakest countries (DE, FR, PL, IT, ES, SE,
   GB), EMA teacher, 40 epochs.

## Layout

```
configs/            final_recipe.json (training recipe), data.json (view + augmentation spec)
src/                model, data pipeline, objectives, and the two training entry points
splits/             official_folds_seed42.csv
artifacts/          per-fold normalization stats and geo-cell assignments read by the pipeline
report/             figure-generation script and the figures used in the write-up
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

```bash
# 1. SSL backbone, one per fold
for f in 0 1 2 3 4; do
  python src/ssl_pretrain_backbone.py --fold $f
done

# 2. retrieval fine-tune across all folds
python src/country_aware_retrieval_finetune.py \
    --config configs/final_recipe.json --folds 0 1 2 3 4

# regenerate report figures from saved predictions
python report/make_figures.py
```

## Formatting

```bash
pip install black isort
isort src report && black src report
```

Configuration is in `pyproject.toml` (black + isort, 88-column).
