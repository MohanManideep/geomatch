# Experiment log

Chronological, honest record of what was tried, what worked, what didn't, and
why. Numbers are pooled 5-fold out-of-fold (OOF) median haversine km on the
11,758 labelled training images unless stated otherwise — the official folds
(`splits/official_folds_seed42.csv`) were used strictly as train-on-4/eval-on-1,
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
  ~35%/43%). Not a recall problem so much as a representation problem for those
  two countries specifically.

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
  This is the strongest evidence that DE/FR location simply isn't encoded in
  the representation -- not a decoding problem.

## 3. SSL pretraining + country-aware finetune (the real encoder win)

- **BYOL self-supervised pretraining** of the backbone only (negative-free,
  momentum target + predictor), 160 epochs, on each fold's training images,
  no labels.
- **Supervised retrieval finetune from that backbone**: same three-view
  architecture, spatial listwise + local listwise + triplet + local-verification
  losses, geographic positives <=50 km, same-country hard negatives mined fresh
  each epoch, DE/FR/PL/IT/ES/SE/GB anchors oversampled, EMA teacher.
- Result (single model, 22-epoch finetune): **57.7 km / 48.8% within 50**,
  down from the locked baseline's 89.16 -- the single biggest win of the project.

## 4. Two more encoder retrains -- both failed

- **Global-view augmentation.** The three-view transform fed the global
  descriptor an unaugmented 512px resize every epoch (only the local crops
  varied), a plausible memorisation source (confirmed by the reranker-gate
  finding above: a fold the encoder trained on reranks to 37 km, the identical
  fold held out reranks to 52 km). Added a random-resized-crop (scale 0.6-1.0)
  on the global view only. **Made it worse** (63.7 vs 53.6 km on a matched
  fold) -- the crop discards geographic context (horizon, skyline, scene
  layout) the descriptor needs; same within-50 rate, much fatter error tail.
- **Finer local tokens.** Local matching used a 4x4 token grid per view;
  raised it to 8x8 (a free architecture change -- `local_projection` is
  grid-size independent, so parameter count is unaffected), plus 48 epochs and
  tighter hard-negative/positive bands. **Made it worse** (61.8 vs 53.6 km),
  overfitting past epoch 36. Finer spatial resolution in the matcher did not
  help distinguish Germany/France scenes.

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
the SSL + country-aware-finetune recipe:

| run / seed | pooled OOF median |
|---|---:|
| seed A | 57.66 km |
| seed B | 59.17 km |
| seed C | 55.84 km |
| **seed D** | **55.60 km** |
| seed E | 57.12 km |
| seed F | 57.63 km |

All cluster in the 55.6-59.2 km band -- i.e. this is the honest, reproducible
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
