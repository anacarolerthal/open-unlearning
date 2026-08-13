# RoAdBlock

**RoAdBlock** (Routed Output-Adapter Difference Blocking) is a lightweight
generation-time unlearning method. It uses a RoAd adapter to identify
forget-specific output evidence and a small prompt router to suppress that
evidence only for matching requests.

## Method

Let the frozen target model produce hidden states $h_t$ and logits

$$
z_0(x,t)=W h_t.
$$

A RoAd-1 adapter $R_\phi$, attached only to the LM head, is trained with
cross-entropy on the forget set $D_F$:

$$
\phi^*=\arg\min_\phi\sum_{(x,y)\in D_F}
\operatorname{CE}(R_\phi z_0(x),y).
$$

RoAd applies a learned rotation and scaling to each pair of vocabulary logits:

$$
R_i=\alpha_i
\begin{bmatrix}
\cos\theta_i&-\sin\theta_i\\
\sin\theta_i& \cos\theta_i
\end{bmatrix}.
$$

The adapter-induced increase

$$
d_\phi(x,t)=R_{\phi^*}z_0(x,t)-z_0(x,t)
$$

acts as a probe for forget-specific evidence. RoAdBlock suppresses only its
positive coordinates:

$$
\boxed{
z_{\mathrm{out}}(x,t)=z_0(x,t)
-\lambda g(x)\,[d_\phi(x,t)]_+
}.
$$

The ordinary router is a GUARD-style MLP over the normalized final prompt-token
hidden state $e(x)$:

$$
g(x)=\mathbb{1}
\left[\sigma \left(w_2^\top
\operatorname{LN}(\operatorname{ReLU}(W_1 e(x)))\right)\geq 0.5\right].
$$

It has one 128-unit hidden layer and is trained on all forget and retain
examples using inverse-frequency-weighted binary cross-entropy. This is a
minimal GUARD-style router, not an exact reproduction of
[GUARD](https://arxiv.org/abs/2505.13312): GUARD uses mean pooling and a much
larger augmented routing dataset, while RoAdBlock uses the final prompt token so
that routing and correction share the target model's single backbone pass.

The finalized TOFU configuration is:

- Llama-3.2-1B-Instruct trained on TOFU-full;
- TOFU forget10 and retain90;
- RoAd-1 on `lm_head`, group size 64;
- adapter learning rate $10^{-3}$ for 10 epochs;
- suppression strength $\lambda=42$;
- GUARD-style MLP threshold 0.5.

Both $z_0$ and $R_\phi z_0$ are computed from the same backbone output.
RoAdBlock therefore requires one backbone pass, preserves the KV cache, and
adds 128,256 RoAd parameters plus 262,657 router parameters. The RoAd transform
itself occupies about 251 KiB in BF16.

## Results

The final sweep trained three independent RoAdBlock instances on TOFU forget10
with seeds 42, 43, and 44.

### Unlearning

| Metric | Mean | Seed range |
|---|---:|---:|
| Forget quality (KS p-value; higher is better) | **0.719** | 0.641--0.758 |
| Forget-quality KS statistic (lower is better) | **0.049** | 0.0475--0.0525 |
| Model utility | **0.591** | 0.590--0.592 |
| Forget ROUGE-L recall | 0.256 | 0.252--0.261 |
| Forget answer probability | 0.0260 | 0.0253--0.0268 |
| Extraction strength | 0.0616 | 0.0615--0.0618 |
| PrivLeak | +48.45 | +48.06--+49.17 |

At the same adapter configuration, the oracle router obtained forget quality
0.758 and utility 0.600. The learned router therefore reaches near-oracle
forgetting while retaining 98.6% of oracle utility. The variation in the KS
p-value is larger than the variation in its underlying statistic, which remains
between 0.0475 and 0.0525.

### Routing

| Metric | Mean across seeds |
|---|---:|
| Balanced accuracy | **91.4%** |
| Forget recall (original and paraphrased average) | **95.0%** |
| Forget-original recall | **100.0%** |
| Forget-paraphrased recall | **90.0%** |
| Holdout positive rate | **7.5%** |
| Real-author positive rate | 11.3% |
| World-fact positive rate | 17.7% |

Compared with the earlier linear, mean-pooled router, the final MLP reduced the
holdout positive rate from 25.75% to 7.5%, increased original-forget recall from
93.5% to 100%, and raised model utility from 0.546 to 0.591.

Raising the routing threshold did not provide a meaningful free improvement.
At threshold 0.9, the macro false-positive rate fell from 12.2% to 9.9%, but
paraphrase recall fell from 90.0% to 86.8%. We therefore retain the natural
threshold of 0.5.

These results support RoAdBlock as a behavioral, safety-oriented intervention.
They do not establish privacy-equivalent data deletion: the remaining PrivLeak
signal must be reported separately from behavioral forgetting.

Results: [W&B sweep `2yaxtuaz`](https://wandb.ai/juanbelieni-lab/open-unlearning/sweeps/2yaxtuaz).

## Continual unlearning

Continual runs keep the frozen decoder activations and one named RoAd adapter per
request in memory. At every stage the router is refit from the fixed retain
activations and replayed forget activations; no previous prompts or checkpoints
are loaded.

Three classifier choices are available to the continual runner:

- `oracle`: use the known request name during evaluation. This is the theoretical
  routing ceiling.
- `guard_multiclass`: refit one balanced 128-unit MLP with classes
  `{retain, request_01, ..., request_t}` and route to the winning request only
  when it beats retain.
- `guard_prototype`: use the same binary GUARD gate, then select the request with
  the highest cosine similarity to its stored KMeans centroids. The default is
  two centroids per request, preserving the two-author structure of TOFU.

Learned routing always applies zero or one adapter per prompt. The correction and
router share a single frozen-backbone pass; adapter composition and learned
soft-routing are intentionally out of scope.
