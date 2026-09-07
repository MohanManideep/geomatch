# Experiment log

Chronological, honest record of what was tried, what worked, what didn't, and
why. Numbers are pooled 5-fold out-of-fold (OOF) median haversine km on the
11,758 labelled training images unless stated otherwise — the 5-fold split we
generated (`splits/folds_seed42.csv`) was used strictly as train-on-4/eval-on-1,
fixed epoch count, no checkpoint selection on the validation fold.

Everything in `archive/` referenced below is kept for reproducibility of these
claims; it is not part of the final submitted pipeline.

## 0. Earlier architecture (abandoned): joint geo-cell classification

A from-scratch RegNetY-400MF with hierarchical geo-cell classification heads
(coarse/a/b/c/fine, decoded via a combinatorial partition decoder) reached
75.16 km median at its best checkpoint (see `archive/docs/V4_LOCAL_RETRIEVAL_RESULTS.md`,
`archive/v2_joint_classification/`). Superseded by the retrieval line below,
which reached lower error with a similar parameter budget.

## 1. Locked baseline: three-view retrieval + local late-interaction

`GeoCPRegNetRetrieval`: shared from-scratch RegNetY-400MF trunk (`weights=None`),
three views per image (global 512px + left/right ~256px crops), multi-scale GeM
pooling to a 384-D descriptor, 48 local tokens (3 views x 4x4 grid, 64-D) for a
ColBERT/MaxSim-style late-interaction rerank. Decode: cosine top-200 shortlist on
the descriptor, rerank by 0.2*global + 0.8*local, take top-1's coordinates.
**4,869,911 trainable parameters.**

- **89.16 km / 45.4% within 50 km** (locked, `archive/.../regnet_cp_v6_locked_oof` snapshot).
- Diagnosis (`archive/retrieval_experiments/src/31_deep_shortlist_recall_audit.py`):
  a <=50 km candidate sits in the retrieval top-200 for ~80% of rows, but the
  decoder only picks it for 45% -- a large ranking gap, worst for DE/FR (oracle@50
  ~35%/43%). Both stages fail there: ranking is the larger loss pooled, but
  ~45% of DE/FR queries retrieve no nearby candidate at all.

## 2. Post-hoc fixes on FROZEN baseline features -- all failed

Every one of these left the encoder untouched and tried to fix ranking at
decode time. None beat the plain blend:

- **Linear metric / residual pairwise reranker** (`27`, `28`, `33` + `configs
  archived`): looked fine on a random held-out training slice, degraded the
  real crossfit (near-duplicate photos in the random slice leak).
- **Honest reranker with an inner-fold holdout** (`37_crossfit_v9_reranker.py`):
  passed its own generalisation gate but STILL degraded the real crossfit
  (+7.9 km) -- root cause found later (section 4).
- **Whitening / query expansion, local-token first-stage**: no measurable gain.
- **Spatial-consensus / kernel-density mode decoding** (`38_v9c_consensus_decode.py`):
  full grid over shortlist depth / kernel bandwidth / temperature; best setting
  was *worse* than the plain blend (72 km vs 57.7 km) -- popular/dense bank
  regions out-vote the true location.
- **Country-gated shortlist / DE-FR bank routing** (`41_routed_decode.py`):
  restricting the shortlist to the query's predicted country. When gated on the
  *true* country this looks great; gated on the *predicted* country (all we
  actually have) it is worse overall (52 -> 56 km), because country prediction
  itself is only ~68% accurate and a wrong gate is catastrophic.
- **Geo-cell classification decode** (`42_geocell_decode.py`): the model's own
  fine-grained classification heads (240 cells) are *just as lost* on DE/FR as
  retrieval (argmax decode: DE 490 km / FR 523 km, vs retrieval's 534/551).
  Two decoders that read the same descriptor fail equally, which suggests a
  shared representation limitation, but does not isolate the cause -- we never
  probed the descriptor directly.

## 3. SSL pretraining + country-aware finetune (the real encoder win)

- **BYOL self-supervised pretraining** of the backbone only (negative-free,
  momentum target + predictor), 160 epochs, on each fold's training images,
  no labels.
- **Supervised retrieval finetune from that backbone**: same three-view
  architecture, spatial listwise + local listwise + triplet + local-verification
  losses, geographic positives <=50 km, same-country hard negatives mined fresh
  each epoch, DE/FR/PL/IT/ES/SE/GB anchors oversampled, EMA teacher.
- Result (single model, 22-epoch finetune): **57.95 km / 48.8% within 50**
  (measured from the per-fold prediction CSVs; an earlier note here said 57.7,
  which is not reproducible from the saved predictions), down from the locked
  baseline's 89.16 -- the single biggest win of the project.

**Which half did the work?** These two changes shipped together, so the -31 km
is not attributable to BYOL alone. A fold-0 comparison separates them (epoch
counts differ, so treat as indicative):

| fold 0 | median |
|---|---:|
| locked baseline (no BYOL, no country-aware), ep 6 | 76.13 km |
| + country-aware finetune from baseline weights, ep 12 | 69.96 km |
| + country-aware from a *v2-joint* start, ep 24 | 71.86 km |
| + BYOL init instead of baseline weights, ep 22 | **53.56 km** |

The runs including BYOL are the lowest by a wide margin, but the conditions
differ in epoch count as well, so no run isolates either change at a matched
schedule and we do not assign a per-component figure.

## 4. Two more encoder retrains -- one backfired, one inconclusive

- **Global-view augmentation.** The three-view transform fed the global
  descriptor an unaugmented 512px resize every epoch (only the local crops
  varied), a plausible memorisation source (confirmed by the reranker-gate
  finding above: a fold the encoder trained on reranks to 37 km, the identical
  fold held out reranks to 52 km). Added a random-resized-crop (scale 0.6-1.0)
  on the global view only. **Worse at every matched epoch on fold 0** (71.9 /
  69.4 / 66.3 / 63.8 / 63.7 km at epochs 20/24/28/30/32, against the shipped
  recipe's 60.2 / 58.3 / 61.5 / 63.4 / 58.1) -- consistent enough to call a
  backfire; same within-50 rate, much fatter error tail.
- **Finer local tokens.** Local matching used a 4x4 token grid per view;
  raised it to 8x8 (a free architecture change -- `local_projection` is
  grid-size independent, so parameter count is unaffected), plus 48 epochs.
  **Inconclusive, not a failure**: at matched epochs on fold 0 it is within
  noise of 4x4 (58.3 vs 60.7 at ep 40; 59.0 vs 61.1 at ep 36; 64.1 vs 60.2 at
  ep 20). Only the extension to 48 epochs drifts up, to 61.8. The earlier
  "it failed" verdict compared its ep-48 endpoint against a *different*
  variant's ep-22 number. 4x4 was kept as the simpler option. Caveat: the
  shipped recipe's own fold-0 median varies by 5.3 km across epochs 20-40, so
  single-fold gaps of that size are not decisive either way.

## 5. Ensembling -- effective, but NOT part of the final submission

Averaging the L2-normalised descriptors and local late-interaction scores of
several *independently seeded* instances of the same recipe reliably improved
the pooled OOF median (pure variance reduction, standard retrieval practice):

| ensemble size | pooled OOF median | within 50 |
|---|---:|---:|
| 1 (any single seed) | 55.6 - 59.2 km | ~49% |
| 3 models | 52.0 km | 49.7% |
| 6 models | 49.65 km | 50.1% |

**This line was abandoned once we re-read the assignment rules**: "your
submitted model must have <=5,000,000 parameters" (singular). Each model here
is individually compliant (4,869,911 params) but an ensemble of N of them is
not one <=5M-parameter model -- it is N times that. The improvement is real
but not a legal submission, so none of these ensembled numbers are the
project's final result. (This mirrors a note already left in
`archive/docs/FINAL_TRAINING.md` from the earlier classification architecture:
"... or ensemble models are used.")

## 6. Final decision

The single-model pooled OOF scores across six independently seeded runs of
the SSL + country-aware-finetune recipe. The three shipped-config runs come
from one command differing only in `--seed-base`:

```bash
python src/country_aware_retrieval_finetune.py \
    --config configs/final_recipe.json --folds 0 1 2 3 4 --seed-base <seed>
```

| seed | config | pooled OOF median |
|---|---|---:|
| 700123 | 34-epoch predecessor | 57.66 km |
| 811777 | 34-epoch predecessor | 59.17 km |
| 933071 | 22-epoch predecessor | 55.84 km |
| **220517** | **shipped, 40 ep** | **55.60 km** |
| 331901 | shipped, 40 ep | 57.12 km |
| 447803 | shipped, 40 ep | 57.71 km |

(The submitted all-data model uses seed 940111, the config default, which is
not among these six.) All cluster in the 55.6-59.2 km band -- i.e. this is the honest, reproducible
performance envelope of one <=5M-parameter, from-scratch, SSL+retrieval model
on this data, independent of random seed. **The best-performing recipe**
(`configs/final_recipe.json`: grid-4 local tokens, 40-epoch finetune, tightened
same-country hard-negative band 60-700 km, geographic positives <=50 km, DE/FR/PL/IT/ES/SE/GB
anchor oversampling) is the final architecture. The submitted model is this
recipe retrained once on **all** 11,758 labelled images (no held-out fold), per
`artifacts/final_full_data/`.

## 7. The unresolved wall: Germany and France

Across every variant, DE and FR sit far behind every other country (typical
median 500+ km vs <200 km for everything else, including similarly-sized PL/
ES/IT which came down to 100-200 km with the same recipe). Evidence points to
a genuine representation gap rather than a decode-time fix:
- retrieval oracle@200 for DE/FR plateaus around 60% (vs 80%+ pooled);
- the classification heads are equally unable to place DE/FR photos (section 2);
- country-aware oversampling and same-country hard-negative mining (both
  explicitly targeting this) moved every other weak country but barely
  touched DE/FR.
Likely cause: Germany and France are large, visually homogeneous at
street level (suburban/rural architecture repeats across huge areas), and a
<=5M-parameter from-scratch encoder at 512px does not have the fine-grained
capacity (or did not get long/varied enough training) to pick out the subtle,
localised cues (signage, plates, micro-architecture) that would disambiguate
them. This is the most promising direction for anyone with more compute.
