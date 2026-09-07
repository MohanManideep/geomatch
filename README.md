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
   geographic positives within 50 km (see note below), same-country hard negatives (60–700 km) re-mined each
   epoch, anchor oversampling for the weakest countries (DE, FR, PL, IT, ES, SE,
   GB), EMA teacher, 40 epochs.

Positive sampling detail: positives are the 8 most descriptor-similar of the
64 geographically nearest that lie within 50 km. For the 10.7% of anchors with
fewer than 8 such neighbours the sampler falls back to the 8 most
descriptor-similar of those same 64 **without the 50 km filter**, so the 50 km
bound holds for ~89% of anchors, not all.

Training time (single RTX 4000 Ada, 20 GB): BYOL backbone ≈ 2 h 40 m per fold;
retrieval finetune ≈ 70 min (40 epochs). The submitted all-data model reuses an
existing backbone, so end to end it is ≈ 70 min plus ≈ 5 min to write
`predictions.csv`.

## Results

Pooled 5-fold out-of-fold error. The shipped-recipe row is the mean ± sample
s.d. of three independently seeded runs (medians 55.60 / 57.12 / 57.71; **best
observed 55.60**). The submitted model retrains the same recipe on all 11,758
images using the config's default seed, which was **not** among the three
evaluated — no seed was selected on evaluation results.

| pooled 5-fold OOF | runs | median | mean | <200 km | <750 km |
|---|---:|---:|---:|---:|---:|
| Geo-cell classification (dropped) | 1 | 75.2 km | — | — | — |
| From-scratch retrieval baseline | 1 | 89.2 km | 618 km | 55.9% | 69.4% |
| + BYOL + country-aware, 22 ep | 1 | 58.0 km | 490 km | 60.7% | 76.2% |
| **+ 40 epochs (shipped recipe)** | 3 | **56.8 ± 1.1 km** | 469 km | 61.4% | 77.1% |
| *(non-compliant)* 3-model ensemble | 1 | 52.0 km | — | — | — |
| *(non-compliant)* 6-model ensemble | 1 | 49.7 km | — | — | — |

The geo-cell predecessor was *better* than the retrieval baseline (75.2 vs
89.2 km) — retrieval started worse and overtook it only after self-supervised
pretraining. The ordering is left non-monotonic rather than hidden.

![Median error by country](images/figures/per_country_median.png)

### Where the error comes from

Splitting every query into *located* (≤50 km), *retrieved but mis-ranked*, and
*never retrieved into the top-200* separates the two failure modes. The last
column is the share of queries with **any** bank photo within 1 km — a measure
of the bank's local geographic coverage. Ranking countries by it reproduces the
performance ordering almost exactly (Spearman −0.89 against median error); that
is an association, not a demonstrated cause, and photos within 1 km need not be
views of the same scene. Single seed (the 55.60 km run).

| | located | mis-ranked | never retrieved | bank <1 km | median |
|---|---:|---:|---:|---:|---:|
| **Pooled** | 49.1% | 30.3% | 20.7% | 30.1% | 55.6 km |
| Iceland | 82.9% | 15.2% | 1.9% | 57.6% | 2.9 km |
| Norway | 65.6% | 23.6% | 10.8% | 49.5% | 9.1 km |
| Finland | 64.8% | 29.3% | 5.9% | 30.1% | 15.0 km |
| Belarus | 62.6% | 29.1% | 8.3% | 40.2% | 26.1 km |
| Turkey | 56.9% | 26.8% | 16.3% | 33.1% | 28.4 km |
| Sweden | 48.5% | 34.6% | 17.0% | 22.8% | 57.2 km |
| United Kingdom | 47.2% | 39.5% | 13.3% | 28.9% | 63.2 km |
| Italy | 43.6% | 27.2% | 29.2% | 28.3% | 214.6 km |
| Spain | 43.4% | 29.5% | 27.1% | 31.8% | 142.2 km |
| Poland | 36.1% | 34.6% | 29.3% | 16.8% | 217.2 km |
| **France** | 25.2% | 32.7% | **42.1%** | 14.5% | **570.4 km** |
| **Germany** | 14.2% | 40.6% | **45.2%** | 9.1% | **504.8 km** |

Ranking is the larger loss pooled, but for Germany and France nearly half of
queries never retrieve a nearby image at all — both stages fail there, so no
reranker could have closed it. Coverage does **not** explain the failure
either: the nearest German bank photo is a median 8.8 km away against a 505 km
German median, so the shortfall is not that no usable neighbour exists.

![Error decomposition by country](images/figures/oracle_gap.png)

## Layout

```
configs/            final_recipe.json (training recipe), data.json (view + augmentation spec)
src/                model, data pipeline, objectives, and the two training entry points
splits/             folds_seed42.csv (the 5-fold split we generated, seed 42)
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

Note: `make_figures.py` reads cached out-of-fold predictions and descriptor
caches under `/var/tmp/luli38se-geomatch/outputs/`, which are training outputs
and are **not** checked into this repository. The committed figures cannot be
reproduced from a clean clone without first re-running the cross-validation
above. `images/geomatch.png` is drawn by hand, not generated.

## Formatting

```bash
pip install black isort
isort src images && black src images
```

Configuration is in `pyproject.toml` (black + isort, 88-column).
