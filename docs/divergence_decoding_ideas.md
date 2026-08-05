# Ideas from Divergence Decoding for the LoRA unlearning experiment

Source: [Divergence Decoding: Inference-Time Unlearning via Auxiliary Models](https://arxiv.org/abs/2605.31293), especially Sections 3, 4.1, 4.5, and 4.8.

These are follow-up experiments, not requirements for the initial two-adapter LoRA implementation.

## 1. Test a one-adapter TOFU variant

The paper uses a full-data auxiliary model $p$ and a retain-only auxiliary model $q$ on TOFU. It avoids a forget-only $p$ because the retain set is much larger, so the difference between forget-only and retain-only fine-tuning may reflect dataset size and fine-tuning dynamics instead of forgetting.

In the current setup, the frozen base model already represents the full-data distribution. It can therefore serve as $p$ without a forget adapter. Only a retain LoRA adapter is needed:

$$
z_U = z_0 + \alpha (z_R - z_0).
$$

With the oracle classifier and $\alpha=1$, evaluation becomes especially simple: use the retain adapter for forget examples and the unchanged base model for every other example. This requires one transformer pass per example and directly tests whether a retain-trained adapter plus a perfect classifier is sufficient.

The existing two-adapter method should remain as the faithful conversion of the original proposal, while this is an ablation.

## 2. Search correction strengths above one

The current sweep tests $\lambda \in \{0.5, 1.0\}$. The paper finds a strong linear-DD region around $\alpha \approx 1.5$ on its benchmarks. Add values such as 1.2, 1.5, and 2.0. These are starting points rather than expected LoRA optima.

Correction strength and classifier choice affect only inference in the current implementation. Adapters should be trained once per training configuration and evaluated under multiple correction/classifier settings instead of being retrained for each setting.

## 3. Add rank-based correction as an ablation

Besides linear logit correction, the paper suppresses the tokens with the largest auxiliary divergence. Its strongest reported region is around top-$k \approx 20$.

Do not set selected logits to $-\infty$: that makes losses infinite and breaks privacy measurements. Follow the paper's finite-loss variant and replace each selected logit with the $k$th largest logit from the unmodified base distribution.

## 4. Distill a successful controller

If inference-time correction works but three transformer passes are too expensive, treat the frozen corrected system as a teacher. Train one LoRA student on forget examples with a temperature-scaled KL objective against the corrected teacher distribution. Retain examples can be included as an explicit preservation term if needed.

This should be attempted only after establishing that the two-adapter/oracle controller provides a useful target.

## 5. Evaluate repeated and adversarial extraction

A dataset-provenance oracle is intentionally unrealistic. Alongside standard TOFU metrics, evaluate paraphrased and indirect prompts and use a repeated-sampling metric such as Leak@K. This distinguishes genuine resistance to extraction from success caused only by exact benchmark routing.

## Compute caveat

LoRA reduces auxiliary parameter storage, but an adapter inserted throughout the transformer does not provide the original head method's single-pass inference. Exact linear correction still evaluates the base, forget-adapted, and retain-adapted networks. Inactive gates should therefore skip adapter passes, and the one-adapter oracle ablation above is particularly valuable.
