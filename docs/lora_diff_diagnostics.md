# LoRA Diff diagnostics and future work

## Current question

Base-referenced one-sided suppression at strength 4 produced the best LoRA Diff
forget quality so far. Lower forget-set ROUGE alone does not explain the result:
strength 6 lowers ROUGE further while moving the truth-ratio distribution far
away from the retain-only reference. The diagnostic experiments below test
whether the strength-4 result comes from a stable forget-specific direction,
the correction magnitude, adapter capacity, or sampling variance.

## Implemented experiments

All diagnostics use base-referenced suppression and evaluate every strength from
the same in-memory adapter. They do not save or reload adapters.

- `sweeps/lora_diff_base_suppression_epoch_diagnostics_tofu_1b.yaml` evaluates
  the same rank-16, seed-42 training trajectory after epochs 1, 3, 5, and 10.
- `sweeps/lora_diff_base_suppression_seed_diagnostics_tofu_1b.yaml` adds seeds
  43 and 44. Together with the epoch sweep's final result, this gives three
  rank-16 seeds without duplicate training.
- `sweeps/lora_diff_base_suppression_rank_diagnostics_tofu_1b.yaml` evaluates
  ranks 1, 2, 4, and 8 at seed 42. The epoch sweep supplies rank 16. LoRA alpha
  equals rank so that alpha/rank remains one and capacity is not confounded with
  PEFT scaling.

Every checkpoint evaluates strengths 2.5 through 5.0 in increments of 0.25.
Each evaluation is written beneath its own directory:

```text
checkpoint-<step>/evals/
  epoch_1_strength_3p75/TOFU_EVAL.json
  epoch_1_strength_4/TOFU_EVAL.json
  ...
```

Final evaluations omit the epoch prefix. W&B keys follow the same labels, for
example `eval/strength_4_forget_quality` and
`eval/epoch_3_strength_4_forget_quality`.

Every strength/epoch evaluation directory is also uploaded as a W&B
`evaluation` artifact. These artifacts contain the complete `TOFU_EVAL.json`
and `TOFU_SUMMARY.json`, including per-example statistics and generated text.
The retain-only `TOFU_EVAL.json` used as the KS and PrivLeak reference is
uploaded once per run as an `evaluation-reference` artifact.

The reduced diagnostic evaluator intentionally excludes retain and general
utility metrics. Oracle routing leaves those examples unchanged, so recomputing
them for every strength provides no new information. A final candidate should
still be run once with the complete TOFU evaluator.

## Saved evidence

The normal evaluator already saves per-example truth ratio, answer probability,
ROUGE, generated text, extraction strength, and MinK membership scores. The
diagnostic evaluator additionally saves, at answer-token prediction positions:

- the fraction and RMS magnitude of positive forget-versus-reference logits;
- mean absolute and RMS correction magnitude;
- the fraction of ground-truth answer tokens with positive divergence;
- mean suppression applied to ground-truth answer tokens;
- the fraction of ground-truth tokens among the top 100 divergent tokens.

Forget quality continues to use the KS-test p-value, while the KS statistic is
also logged as an effect size. A high p-value without a consistently small KS
statistic across seeds should not be treated as evidence of distributional
matching.

The final seed-42 epoch run additionally loads the official retain90 model once
for analysis only. `diagnostics/retain_direction.json` records the cosine and
norm ratio between the candidate and ideal retain-model logit directions,
top-100 suppression-token overlap, and `KL(retain || corrected)` at every
tested strength. Directional comparisons center each vocabulary-logit vector so
that behaviorally irrelevant constant offsets do not affect the result. The
retain model is not required by the method or its normal evaluation path.
The complete `retain_direction.json` is uploaded separately as an
`evaluation-diagnostics` artifact.

## Decisions enabled by the diagnostics

1. **Stable strength region.** Strength 4 is credible only if neighboring
   strengths and multiple seeds also approach the retain truth-ratio
   distribution. A single isolated p-value is not sufficient.
2. **Direction versus magnitude.** Compare the best strength with the positive
   divergence RMS at epochs 1, 3, 5, and 10. If their product stays roughly
   constant, training learns the useful direction early and mainly changes its
   scale afterward.
3. **Minimum useful rank.** If ranks 1--4 preserve the distributional match and
   ground-truth-token coverage, all-linear rank-16 LoRA is unnecessary.
4. **Selective attribution.** A useful forget probe should place true answer
   tokens among its largest positive divergences. Broad correction mass with
   low target-token coverage would instead indicate indiscriminate suppression.
5. **Generation quality.** Low ROUGE must be checked against the saved text.
   Refusal, plausible uncertainty, incorrect facts, and gibberish all reduce
   ROUGE but represent different behaviors.

The primary selection criteria should be consistency across seeds, a small KS
statistic, PrivLeak near zero, and reduced extraction. ROUGE is supporting
evidence rather than the optimization target.

## Future method changes

These ideas are deliberately not part of the diagnostic implementation.

### One-pass residual patch

Replace the two-pass logit difference with a rank-1 to rank-4 residual patch at
one late hidden state:

\[
h'_t = h_t + g(x) U\,\sigma(Vh_t), \qquad z_t = W_{LM}h'_t.
\]

The frozen backbone would run once with its normal KV cache. The patch could be
trained directly with NPO or distilled from the best diagnostic correction. A
direct-NPO failure followed by successful distillation would indicate an
optimization problem; failure of both would indicate insufficient patch
capacity or a poor intervention layer.

### Lightweight routing

After establishing the oracle upper bound, replace it with a prototype or
linear classifier over a frozen prompt representation. Compute the gate once
during prefill and reuse it for generation. Report false-positive damage and
end-to-end unlearning at every threshold rather than only router AUROC. Nearest
prototypes, gate scores, and the tokens most changed by the patch provide a
simple explanation for each intervention.

### Continual unlearning

Split forget10 into sequential requests and evaluate every previous request
after each update. Begin with one rank-one direction and one routing prototype
per request. This gives transparent storage growth and avoids interference
between requests. Only consider shared-basis compression after this simple bank
has established the quality and storage trade-off.

The comparison should include O3 and LUNAR and report trainable parameters,
stored bytes per request, backbone passes, KV-cache compatibility, latency,
current-request forgetting, previous-request forgetting, retain utility, and
router false positives.
