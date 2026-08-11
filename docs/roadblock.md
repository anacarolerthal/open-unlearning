# RoAdBlock

**RoAdBlock** (Routed Output-Adapter Difference Blocking) is a one-backbone-pass
unlearning intervention. It uses a tiny RoAd adapter as a probe for memorized
knowledge, then suppresses the evidence exposed by that probe.

## Method

Let the frozen target model produce

$$
z_0(x,t)=W h_\theta(x_{\leq t}),
$$

where the target model was trained on the forget data. A RoAd-1 transform
$R_\phi$ is attached only to the language-model output and trained with
cross-entropy on the forget set:

$$
\phi^*=\arg\min_\phi\sum_{(x,y)\in D_F}
\operatorname{CE}(R_\phi z_0(x),y).
$$

RoAd is a block-diagonal rotation and scaling of the vocabulary logits. For
each paired coordinate,

$$
R_i=\alpha_i
\begin{bmatrix}
\cos\theta_i&-\sin\theta_i\\
\sin\theta_i& \cos\theta_i
\end{bmatrix}.
$$

The trained adapter exposes positive forget-specific evidence

$$
d_\phi(x,t)=R_{\phi^*}z_0(x,t)-z_0(x,t).
$$

Given a router $g(x)\in\{0,1\}$, inference applies one-sided suppression:

$$
\boxed{\quad z_{\text{out}}(x,t)=z_0(x,t)
-\lambda g(x)\,[d_\phi(x,t)]_+\quad}.
$$

The confirmed experiment used an oracle router: $g(x)=1$ for forget examples
and zero otherwise. The implementation now supports four router MVPs:

$$
g(x)=\begin{cases}
\mathbb{1}[x\in D_F] & \text{oracle},\\
1 & \text{no classifier},\\
\mathbb{1}[\sigma(w^\top e(x)+b)\geq 0.5] & \text{GUARD-style},\\
\mathbb{1}[\max_{f\in D_F}\cos(e(x),e(f))\geq\delta] & \text{CURaTE-style}.
\end{cases}
$$

Here $e(x)$ is the mean-pooled final hidden state of the prompt. The GUARD-style
router trains a linear head on balanced forget and retain embeddings. The
CURaTE-style router stores forget embeddings and calibrates $\delta$ on balanced
forget and retain examples. These are deliberately minimal variants, not exact
reproductions of [GUARD](https://arxiv.org/abs/2505.13312) or
[CURaTE](https://arxiv.org/abs/2604.14644).

The confirmed adapter configuration is RoAd-1, learning rate $10^{-3}$, 10
epochs, and $\lambda=42$.

Both $z_0$ and $R_\phi z_0$ come from the same backbone output. RoAdBlock
therefore needs one backbone pass, preserves the KV cache, and stores 128,256
trainable parameters for Llama-3.2-1B (about 251 KiB in BF16).

## Results

The confirmation sweep used TOFU forget10 and three independently trained
adapters. All three selected $\lambda=42$.

| Metric | Mean | Seed range |
|---|---:|---:|
| Forget quality (KS p-value) | 0.719 | 0.641--0.758 |
| KS statistic (lower is better) | 0.049 | 0.0475--0.0525 |
| Forget ROUGE-L recall | 0.256 | 0.252--0.261 |
| Forget answer probability | 0.0260 | 0.0253--0.0268 |
| Extraction strength | 0.0616 | 0.0615--0.0618 |
| PrivLeak | +53.8 | +53.5--+54.3 |

The forget-quality result is substantially stronger and more stable than the
tested output-only LM-head LoRA (best p-value 0.000775) and all-linear LoRA Diff
(mean best p-value 0.193). It also exceeds the reproduced NPO forget-quality
p-value of 0.02, although the oracle router makes this an upper-bound experiment
rather than a deployable comparison.

The reduced sweep measured forget behavior only. For oracle-negative inputs,
$g(x)=0$ makes the output exactly equal to the target model. Privacy remains
the main weakness: RoAdBlock changes or obscures answers effectively, but does
not yet match the retain model under membership-inference metrics.

Results: [W&B sweep `38o2opf7`](https://wandb.ai/juanbelieni-lab/open-unlearning/sweeps/38o2opf7).

## Before continual unlearning

1. **Validate routing.** Compare the router MVPs on false positives, paraphrase
   recall, threshold calibration, latency, and adversarial bypasses.
2. **Validate the candidate.** Run the complete TOFU evaluation once, repeat on
   other forget splits and models, and report privacy alongside behavioral
   forgetting.
3. **Build an adapter bank.** Store one RoAd transform and router key per request;
   define routing when zero, one, or several requests match the same prompt.
4. **Test sequentially.** Add forget requests one at a time and re-evaluate every
   previous request, retain utility, router errors, latency, and bytes per request.
5. **Compare continual baselines.** Match O3 and related methods on quality while
   reporting the advantages RoAdBlock targets directly: one backbone pass,
   KV-cache compatibility, transparent routing, and small per-request state.
