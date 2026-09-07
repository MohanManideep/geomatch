# Every experiment: result and issue

One entry per attempt, in chronological order. All "result" numbers are
**pooled 5-fold out-of-fold (OOF) median haversine km** on the 11,758 labelled
training images (train on 4 official folds, evaluate on the held-out one,
fixed epoch count, no checkpoint selection on the eval fold) unless noted.
`EXPERIMENT_LOG.md` has the fuller narrative version of the same material.

---

### 1. Hierarchical geo-cell classification (abandoned architecture)
- **Experiment:** from-scratch RegNetY-400MF, three-view input, six
  classification heads (country + coarse/a/b/c/fine geo-cells), decoded via a
  combinatorial partition decoder (best cell combination -> centroid).
- **Result:** 75.16 km median at the best checkpoint.
- **Issue:** classification decoding is bounded by the cell resolution.
  Retrieval (next section) is also discrete -- it can only return one of the
  11,758 training coordinates -- but that is a much finer grid: the
  geographically nearest training photo to a held-out photo is a median 3.29 km
  away and within 50 km for 99.8% of rows, so the discretisation is nowhere
  near binding. Retrieval reached lower error at a similar parameter budget,
  so this line was dropped.

### 2. Three-view retrieval + local late-interaction (locked baseline)
- **Experiment:** same shared from-scratch backbone, but descriptors instead
  of classes: 384-D global descriptor (GeM pooling) + 48 local tokens (3
  views x 4x4 grid, 64-D) for a ColBERT/MaxSim-style rerank. Decode: cosine
  top-200 shortlist, rerank by 0.2*global + 0.8*local, take top-1.
- **Result:** **89.16 km / 45.4% within 50 km.**
- **Issue:** diagnosed with a shortlist-depth audit: a <=50 km candidate is in
  the top-200 for ~80% of rows, but the decoder only picks it ~45% of the
  time. Ranking is the larger loss pooled -- but ~20% of rows never retrieve a
  nearby candidate at all, and in Germany/France that recall failure is ~45%,
  so it is *both* a ranking and a recall problem, not purely the former.

### 3. Linear metric / MLP candidate ranker on frozen baseline descriptors
- **Experiment:** learn a small linear or MLP scorer over the top-K shortlist
  using engineered features (rank, similarity gap, geo-density), on top of
  the frozen, already-trained baseline encoder.
- **Result:** looked good on a random held-out slice of training rows.
- **Issue:** the random slice contains near-duplicate photos of the same
  scene, so it leaks; the same reranker made the real crossfit worse.

### 4. Reranker with an honest inner-fold holdout gate
- **Experiment:** fixed #3's leakage by holding out an entire *inner* official
  fold from the reranker's own training data, so the gate can't leak via
  near-duplicates.
- **Result:** the gate passed (looked like a real +5 km improvement).
- **Issue:** the real 5-fold crossfit still got *worse* (+7.9 km). Root cause
  found later: the gate's probe fold was held out from the reranker but not
  from the *encoder* -- the encoder had already trained on 4/5 of the probe
  fold's neighbourhood, so the gate measured an easier problem than the real
  held-out-encoder case.

### 5. Whitening / query expansion / local-token first-stage retrieval
- **Experiment:** three more frozen-feature decode tweaks: descriptor
  whitening + query expansion, and using local tokens as the *first* retrieval
  stage instead of a rerank.
- **Result:** no measurable improvement over the plain blend.
- **Issue:** all of these still only rearrange the same frozen ranking; they
  don't add new information the encoder didn't already put in the descriptor.

### 6. Spatial-consensus / kernel-density mode decoding
- **Experiment:** instead of trusting the single top-scoring candidate,
  score each shortlist candidate by how much retrieval-confidence-weighted
  "mass" from other candidates piles up within a target-radius kernel around
  it (a mean-shift-style mode estimate) -- swept shortlist depth, kernel
  bandwidth, and temperature.
- **Result:** best setting found was **72 km**, worse than the 57.7 km
  plain blend at the time.
- **Issue:** the bank isn't uniformly distributed -- popular/dense regions
  (big cities, tourist spots) out-vote the true, sparser location.

### 7. Country-gated shortlist / DE-FR bank routing
- **Experiment:** restrict the retrieval shortlist to same-country (or
  DE+FR-only) bank candidates, gated on the model's own country prediction.
- **Result:** gated on the *true* country, this looks great (a DE/FR-only
  bank roughly halves their median error). Gated on the *predicted* country
  (all that's actually available at test time), pooled median got *worse*
  (52 -> 56 km).
- **Issue:** country prediction itself is only ~68% accurate, and a wrong
  gate forces the answer into entirely the wrong country -- the damage from
  misroutes outweighs the gain from correct routes.

### 8. Geo-cell classification decode (reusing the earlier classification idea on the SSL-finetuned encoder)
- **Experiment:** decode from the model's own (otherwise unused) 240-cell
  classification head instead of retrieval, to see if classification handles
  DE/FR better than nearest-neighbour matching.
- **Result:** DE 490 km / FR 523 km -- **barely better** than retrieval's
  534/551 km, i.e. still catastrophic.
- **Issue:** two decoders reading the same descriptor fail equally, which
  locates the problem upstream of the decoder. It does not establish *what*
  the descriptor is missing -- we never probed that directly.

### 9. BYOL self-supervised pretraining + country-aware retrieval finetune
- **Experiment:** pretrain the backbone with BYOL (negative-free
  self-supervised learning, momentum target + predictor) on each fold's
  training images, 160 epochs, no labels; then finetune the full retrieval
  architecture from that backbone with country-aware hard-negative mining
  (same-country, visually-similar negatives mined fresh every epoch) and
  anchor oversampling for the weakest countries (DE/FR/PL/IT/ES/SE/GB).
- **Result:** **57.95 km / 48.8% within 50 km** (measured from the per-fold
  prediction CSVs; an earlier note in this repo said 57.7, which is not
  reproducible from the saved predictions) -- the single biggest win of the
  project, -31 km vs. the locked baseline.
- **Issue:** two changes shipped together, so the -31 km is not attributable
  to BYOL alone. Fold-0 separation: baseline 76.13 -> country-aware only 69.96
  (from baseline weights, ep 12) / 71.86 (from a joint start, ep 24) -> with
  BYOL init 53.56 (ep 22). Country-aware buys ~6 km, BYOL ~16 km more.
  Also: DE/FR barely moved (still 500+ km); every other country improved.

### 10. Anti-memorisation augmentation on the global view
- **Experiment:** the three-view transform fed the global descriptor an
  unaugmented 512px resize every training epoch (only the two local crops
  varied). Suspecting encoder memorisation (confirmed indirectly by
  experiment #4's finding: an encoder-seen fold reranks to 37 km, the same
  fold unseen reranks to 52 km), added a random-resized-crop (scale 0.6-1.0)
  on the global view only, plus more weight decay and a higher EMA decay.
- **Result:** **worse at every matched epoch on fold 0** -- 71.9 / 69.4 / 66.3
  / 63.8 / 63.7 km at epochs 20/24/28/30/32 against the shipped recipe's
  60.2 / 58.3 / 61.5 / 63.4 / 58.1, at the same within-50 rate (fatter error
  tail, not a ranking change).
- **Issue:** cropping the global view discards geographic context (horizon,
  skyline, overall scene layout) the descriptor actually needs; the fix for
  memorisation cost more than it saved.

### 11. Finer local-matching tokens + longer training
- **Experiment:** raised the local-token grid from 4x4 to 8x8 per view (a
  free architecture change -- the token projection layer is grid-size
  independent, so parameter count is unaffected), 48 epochs instead of 22,
  tighter hard-negative/positive distance bands.
- **Result:** **inconclusive, not a failure.** At matched epochs on fold 0 the
  8x8 grid is within noise of the shipped 4x4 (58.3 vs 60.7 km at epoch 40;
  64.1 vs 60.2 at epoch 20; 59.0 vs 61.1 at epoch 36). Only the extension to
  48 epochs drifts back up, to 61.8 km.
- **Issue:** the earlier "8x8 failed" verdict compared its epoch-48 endpoint
  against a *different* variant's epoch-22 number -- an unfair comparison. The
  honest reading: finer tokens bought nothing measurable, and training past
  ~40 epochs drifts upward. 4x4 was kept as the simpler option. Note the
  shipped recipe's own fold-0 median wanders between 58.1 and 63.4 km over
  epochs 20-40, so single-fold gaps under ~3 km carry no signal.

### 12. Ensembling multiple independently-seeded models
- **Experiment:** L2-normalise and average the descriptors and local
  late-interaction scores of several separately-trained instances of the
  same recipe (different random seeds), before the usual blend+top-1 decode.
- **Result:** reliable, standard variance-reduction effect --
  3 models: **52.0 km**; 6 models: **49.65 km / 50.1% within 50**.
- **Issue:** **not usable.** The assignment requires "your submitted model"
  to have <=5,000,000 parameters (singular). Each model here is individually
  compliant (4,869,911 params) but N of them combined is not one <=5M model.
  The improvement is real but this is not a legal submission -- dropped once
  the rules were re-read carefully. (Also visible per-model: none of the
  individual seeds beat 55.6 km alone -- the gain is purely from averaging.)

### 13. Final decision: one model, the shipped recipe
- **Experiment:** compare independently-seeded single-model runs on their own,
  un-ensembled, pooled OOF score.
- **Result:** three seeds of the **exact shipped config** give 55.60 / 57.12 /
  57.71 km -> mean **56.8 +/- 1.1**, median 57.12, best observed 55.60. Three
  further runs of *earlier* configs in the same family (22-epoch, and two
  variants) give 57.66 / 59.17 / 55.84, so the whole family spans 55.6-59.2 km.
  Quoting 55.60 alone would be cherry-picking the best of three; the mean is
  the honest headline. Seed-to-seed spread (~1.1 km) is larger than the
  difference between the 22- and 40-epoch recipes, so the extra epochs are not
  a demonstrated improvement.
- **Decision:** ship the recipe (grid-4 local tokens,
  40-epoch finetune, same-country hard-negative band 60-700 km, geographic
  positives <=50 km, DE/FR/PL/IT/ES/SE/GB anchor oversampling), retrained once on
  **all** 11,758 labelled images (no held-out fold) for the actual
  submission.

---

## Summary

| stage | pooled OOF median | evidence | notable for |
|---|---:|---|---|
| Geo-cell classification (abandoned architecture) | 75.16 km | 5-fold | why classification decoding was dropped |
| **Locked baseline (retrieval + late-interaction)** | **89.16 km** | 5-fold | starting point; the error decomposition |
| SSL + country-aware finetune, 22 ep | 57.95 km | 5-fold | the one large win (-31 km) |
| **Shipped recipe, 40 ep** | **56.8 ± 1.1 km** (3 seeds) | 5-fold ×3 | what's submitted; best seed 55.60 |
| Crop-augmentation variant | worse at every matched epoch (63.7 vs 58.1 @32) | fold 0 only | a plausible fix that backfired |
| Finer-token (8×8) variant | within noise (58.3 vs 60.7 @40) | fold 0 only | bought nothing; 48 ep drifts to 61.8 |
| Ensembles (3/6 models) | 52.0 / 49.65 km | 5-fold | effective but **not compliant** -- excluded |

Error decomposition of the shipped recipe (pooled OOF): 49.1% located within
50 km, 30.3% retrieved-but-mis-ranked, 20.7% never retrieved into the top-200.
For Germany those are 14.2 / 40.6 / 45.2 -- i.e. both recall and ranking fail
there, so no reranking fix could have closed that gap.
