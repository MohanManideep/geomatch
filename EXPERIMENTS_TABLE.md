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
- **Issue:** classification-only decoding quantizes location to a fixed cell
  grid; can't get closer than the cell resolution. Retrieval (next section)
  gives a continuous coordinate and reached lower error at a similar
  parameter budget, so this line was dropped.

### 2. Three-view retrieval + local late-interaction (locked baseline)
- **Experiment:** same shared from-scratch backbone, but descriptors instead
  of classes: 384-D global descriptor (GeM pooling) + 48 local tokens (3
  views x 4x4 grid, 64-D) for a ColBERT/MaxSim-style rerank. Decode: cosine
  top-200 shortlist, rerank by 0.2*global + 0.8*local, take top-1.
- **Result:** **89.16 km / 45.4% within 50 km.**
- **Issue:** diagnosed with a shortlist-depth audit: a <=50 km candidate is in
  the top-200 for ~80% of rows, but the decoder only picks it ~45% of the
  time -- a ranking problem, not a recall problem, worst for Germany/France.

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
- **Issue:** this is the strongest evidence that DE/FR location is a
  genuine representation gap, not a decoding-method problem -- neither
  decoding strategy can find information the encoder didn't learn.

### 9. BYOL self-supervised pretraining + country-aware retrieval finetune
- **Experiment:** pretrain the backbone with BYOL (negative-free
  self-supervised learning, momentum target + predictor) on each fold's
  training images, 160 epochs, no labels; then finetune the full retrieval
  architecture from that backbone with country-aware hard-negative mining
  (same-country, visually-similar negatives mined fresh every epoch) and
  anchor oversampling for the weakest countries (DE/FR/PL/IT/ES/SE/GB).
- **Result:** **57.7 km / 48.8% within 50 km** -- the single biggest win of
  the project (-31 km vs. the locked baseline).
- **Issue:** DE/FR barely moved (still 500+ km); every other country improved
  substantially.

### 10. Anti-memorisation augmentation on the global view
- **Experiment:** the three-view transform fed the global descriptor an
  unaugmented 512px resize every training epoch (only the two local crops
  varied). Suspecting encoder memorisation (confirmed indirectly by
  experiment #4's finding: an encoder-seen fold reranks to 37 km, the same
  fold unseen reranks to 52 km), added a random-resized-crop (scale 0.6-1.0)
  on the global view only, plus more weight decay and a higher EMA decay.
- **Result:** **worse** -- 63.7 km vs. 53.6 km on a matched fold, at the same
  within-50 rate (fatter error tail, not a ranking change).
- **Issue:** cropping the global view discards geographic context (horizon,
  skyline, overall scene layout) the descriptor actually needs; the fix for
  memorisation cost more than it saved.

### 11. Finer local-matching tokens + longer training
- **Experiment:** raised the local-token grid from 4x4 to 8x8 per view (a
  free architecture change -- the token projection layer is grid-size
  independent, so parameter count is unaffected), 48 epochs instead of 22,
  tighter hard-negative/positive distance bands.
- **Result:** **worse** -- 61.8 km vs. 53.6 km on a matched fold, and the
  validation curve visibly overfits past epoch 36 (dips to ~59 km around
  epoch 30, drifts back up to 61.8 by epoch 48).
- **Issue:** finer spatial resolution in the local matcher did not help
  distinguish Germany/France street scenes from each other -- more evidence
  the bottleneck is representational, not resolution.

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
- **Experiment:** compare six independently-seeded single-model runs from
  the SSL + country-aware-finetune family (experiments #9 and #12) on their
  own, un-ensembled, pooled OOF score.
- **Result:** 57.66 / 59.17 / 55.84 / **55.60** / 57.12 / 57.63 km -- all six
  land in a tight 55.6-59.2 km band regardless of seed. This is the honest,
  reproducible performance envelope of one <=5M-parameter, from-scratch,
  SSL-pretrained retrieval model on this data.
- **Decision:** ship the best-performing single recipe (grid-4 local tokens,
  40-epoch finetune, hard-negative band 40-400 km, geographic positives
  <=35 km, DE/FR/PL/IT/ES/SE/GB anchor oversampling), retrained once on
  **all** 11,758 labelled images (no held-out fold) for the actual
  submission.

---

## Summary

| stage | pooled OOF median | notable for |
|---|---:|---|
| Geo-cell classification (abandoned architecture) | 75.16 km | why classification-only decoding was dropped |
| **Locked baseline (retrieval + late-interaction)** | **89.16 km** | starting point; the ranking-gap diagnosis |
| SSL + country-aware finetune | 57.66 km | the one big architectural win |
| Crop-augmentation variant | worse (fold: 63.7 vs 53.6) | a plausible fix that backfired |
| Finer-token / longer-training variant | worse (fold: 61.8 vs 53.6) | overfitting, capacity isn't the bottleneck |
| Ensembles (3/6 models) | 52.0 / 49.65 km | effective but **not compliant** -- excluded |
| **Final single-model recipe** | **55.60 km** (best seed) | what's actually submitted |
