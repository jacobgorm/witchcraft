# XTR Training HOWTO

## Background

XTR (Token-level Retrieval) replaces ColBERT's brute-force MaxSim scoring with
centroid-based approximate token retrieval (k-means clustering + top-k centroid
probing). This requires embeddings that survive the centroid approximation well.

Google's original XTR weights (T5-based) transfer directly to Warp's centroid
retrieval without retraining. ColModernVBert, trained with ColBERT loss, scores
well with brute-force MaxSim (0.34 NDCG@10 on NFCorpus) but loses ~26% under
centroid retrieval (0.25). The goal of XTR training is to close this gap.

## XTR Paper Recipe (Lee et al., 2024)

From the paper: "Following Ni et al. [2021], we fine-tune XTR on MS MARCO with
a fixed set of hard negatives from RocketQA [Qu et al., 2021]."

Key parameters:
- **Data**: MSMARCO with RocketQA hard negatives (model-mined, not BM25)
- **Loss**: XTR (Eq. 2) — in-batch top-k token retrieval, Z-normalization
- **k_train**: 128 (tokens retrieved per query token during training)
- **batch_size**: 320 (in-batch negatives)
- **Temperature**: GTR uses τ=0.01 with cosine similarity; XTR follows GTR.
  For L2-normalized embeddings, temperature ~0.05 is needed for meaningful
  cross-entropy gradients.
- **MSE imputation**: inference-only, not used during training

**Important**: The paper trains T5 from scratch for XTR. This is fundamentally
different from finetuning an already-trained ColBERT model.

## Results Summary (NFCorpus NDCG@10)

| Configuration | Python MaxSim | Python XTR | Warp |
|---|---|---|---|
| Pretrained ColModernVBert (baseline) | 0.340 | — | 0.250 |
| Full model, XTR loss, temp=1.0 | degrading | degrading | 0.168→0.097 |
| Frozen encoder, XTR loss, temp=1.0 | 0.341 | 0.324 | 0.271 (step 24068) |
| Frozen encoder, XTR loss, temp=0.05 | 0.341 | 0.335 | 0.255 (epoch 5) |
| XTR-base (Google, from paper) | — | — | ~0.33 (T5-based) |
| Granite 97M v23 best | — | — | 0.284 |
| Granite 311M v1 best | — | — | 0.322 |
| Granite English-r2 baseline | — | — | 0.359 |

## Granite 97M Run Ledger (Local Rust-Warp, May 2026)

Fixed-pipeline Granite baseline: **0.2739169018116819** NFCorpus NDCG@10.

| Run | Main config change | Checkpoint | NFCorpus NDCG@10 |
|---|---|---:|---:|
| v26 | 97M random-projection baseline | step_0 | 0.21947405340869144 |
| v13 | encoder frozen 500 steps, projection_lr=1e-3, KD enabled | step_600 | 0.2430429139004257 |
| v13 | same | step_800 | 0.25213246547356516 |
| v14 | no encoder freeze, no projection_lr, KD enabled | step_200 | 0.2252101502348956 |
| v14 | same | step_400 | 0.2543497803435053 |
| v15 | no encoder freeze, projection_lr=1e-3 restored, KD enabled | step_200 | 0.2328396969295673 |
| v15 | same | step_400 | 0.2651908886665361 |
| v15 | same | step_600 | 0.268679778870787 |
| v15 | same | step_800 | 0.2693522752702339 |
| v15 | same | step_1200 | **0.2749987621498651** |
| v15 | same | step_1400 | 0.27469069260708573 |
| v15 | same | step_1600 | 0.27362779110040253 |
| v16 | v15 optimizer shape, KD removed | step_200 | 0.23844882371964335 |
| v16 | same | step_400 | 0.25414287549026837 |
| v16 | same | step_600 | 0.25222341798335174 |
| v17 | v15 plus kd_k_train=64 (XTR KD scoring) | step_200 | 0.23372743704327087 |
| v17 | same | step_400 | 0.20341501406705112 |
| v18 | v15 plus warmup_steps=1000 | step_200 | 0.21318167514692057 |
| v18 | same | step_400 | 0.2348510332755326 |
| v18 | same | step_600 | 0.251956892016143 |
| v19 | v15 plus kd_weight=0.75 | step_200 | 0.2241256401532028 |
| v19 | same | step_400 | 0.2538776982223963 |
| v20 | v15 plus encoder_lr=1e-5 | step_200 | 0.22738367500898707 |
| v20 | same | step_400 | 0.25168886277976976 |
| v21 | v15 plus k_train=128 | step_200 | 0.22509175772168102 |
| v21 | same | step_400 | 0.2654333851072553 |
| v21 | same | step_600 | 0.2607487013937238 |
| v22 | v15 plus projection_lr=2e-3 | step_200 | 0.23488606956858177 |
| v22 | same | step_400 | 0.25791922958745694 |
| v23 | v15 plus kd_weight=0.25 | step_200 | 0.24794525168900397 |
| v23 | same | step_400 | 0.2672298303902081 |
| v23 | same | step_600 | 0.267128798989571 |
| v23 | same | step_800 | 0.2701495270518326 |
| v23 | same | step_1000 | 0.27363121240230565 |
| v23 | same | step_1200 | 0.2764209026902891 |
| v23 | same | step_1400 | **0.28386522818511634** |
| v23 | same | step_1600 | 0.27752483944114764 |
| v24 | v23 but kd_weight=0.125 | step_200 | 0.23135695786662566 |
| v25 | v23 but kd_weight=0.375 | step_200 | 0.23675041692356819 |
| v26 | v23 hyperparameter replicate | step_200 | 0.24425351124076639 |
| v26 | same | step_400 | 0.2579981774024475 |

Interpretation:
- The v13 projection-only warmstart hurt Granite. Even after unfreeze, v13 lagged.
- v14 showed that no-freeze was viable, but it confounded the test by starving
  the random projection head at lr=2e-5.
- v15 is the cleanest run so far: train encoder from step 0, keep
  projection_lr=1e-3. It beat the fixed Granite baseline at step_1200, then
  flattened/slightly regressed.
- v13's headline train loss is not comparable to v11 pure contrastive loss:
  v13 total loss is 0.5 * c_loss + 0.5 * kd_loss, and the KD term is much
  smaller/smoother.
- v16 no-KD underperformed v15 by step_400/step_600, so KD is useful for
  Granite; the next question is KD scoring/form rather than whether to drop KD.
- v17 KD-XTR at kd_weight=0.5 was unstable/destructive by step_400. The KD-XTR
  loss scale was much larger and spikier than exact KD; do not use this setting
  as-is.
- v18 longer warmup lagged badly through step_600; Granite does not appear to
  benefit from matching mmBERT's 1000-step warmup under this setup.
- v19 stronger exact KD underperformed v15 by step_400; kd_weight=0.5 remains
  the best exact-KD balance so far.
- v20 lower encoder LR underperformed v15 by step_400; encoder_lr=2e-5 remains
  better than 1e-5 for this no-freeze setup.
- v21 matched/slightly beat v15 at step_400 but fell behind by step_600;
  k_train=128 is not the next lever unless later evidence contradicts this.
- v22's higher projection LR gave a small step_200 bump but underperformed badly
  by step_400; projection_lr=1e-3 remains the safer setting.
- v23 lighter exact KD is the new best branch. It improved early learning and
  reached a much higher peak at step_1400 before dropping by step_1600, so the
  best checkpoint is checkpoints-granite-97m-v23/step_1400.
- v24 reduced exact KD too far. It was far behind v23 at step_200, so the run
  was stopped early.
- v25 increased KD from v23 toward the old v15 weight and also lagged badly at
  step_200. The best tested KD weight is now bracketed at 0.25.
- v26 reran the v23 hyperparameters because training is not explicitly seeded.
  It did not reproduce v23's early trajectory and was stopped at step_400; keep
  v23 step_1400 as the best checkpoint, but treat the jump as stochastic until
  more replicas or seeded training confirm it.

## Granite 311M Run Ledger (Local Rust-Warp, May 2026)

Config: `config-granite-311m.yaml`, using
`ibm-granite/granite-embedding-311m-multilingual-r2` with the v23/v26 recipe:
encoder_lr=2e-5, projection_lr=1e-3, k_train=64, exact KD, kd_weight=0.25,
warmup_steps=200, batch_size=128, gradient_accumulation_steps=2.

| Run | Main config change | Checkpoint | NFCorpus NDCG@10 |
|---|---|---:|---:|
| v1 | 311M multilingual baseline | step_0 | 0.2613236203511852 |
| v1 | 311M base with v23/v26 recipe | step_200 | 0.2829528424111914 |
| v1 | same | step_400 | 0.2859617313919183 |
| v1 | same | step_600 | 0.30170769699604694 |
| v1 | same | step_800 | 0.30580443389472123 |
| v1 | same | step_1000 | 0.30918259859668706 |
| v1 | same | step_1200 | 0.3089188714676717 |
| v1 | same | step_1400 | 0.3134492342515246 |
| v1 | same | step_1600 | **0.3221250963192639** |
| v1 | same | step_1800 | 0.31575133052469273 |
| v1 | same | step_2000 | 0.318479753581584 |
| v1 | same | step_2200 | 0.31670919608219356 |
| v2-qat | resume v1 step_1600, QAT enabled | step_200 | 0.3163281317280162 |
| v3-top2-sigreg | 311M multilingual, encoder frozen except top 2 layers, no KD, `sigreg_weight=1e-3` | step_50 | 0.2293834773776554 |
| v3-top2-sigreg | same | step_100 | 0.23797666068017345 |
| v3-top2-sigreg | same | step_400 | 0.25538407555198145 |

Interpretation:
- The 311M base materially improves over 97M under the same recipe, moving the
  best Warp NFCorpus score from 0.284 to 0.322.
- The run peaked at `checkpoints-granite-311m-v1/step_1600`; three later
  checkpoints stayed below the peak, so the GPU run was stopped at step_2200.
- This nearly reaches the T5-based target (~0.33) without QAT or additional
  Warp-side tuning. The next experiment should change one variable at a time
  from this 311M baseline.
- QAT continuation from the best checkpoint underperformed after 200 optimizer
  steps and was stopped early. The current QAT wrapper only simulates Linear
  layer quantization, not embedding-table quantization, so it is not a complete
  match for the Rust Q4 export path.
- The top-2 unfreeze + weak SIGREG ablation did not help. It recovered from
  0.229 at step_50 to 0.255 at step_400, but remained below the untrained
  311M multilingual baseline of 0.261. This rejects that specific constrained
  adaptation recipe; it does not rule out SIGREG with a stronger/full-encoder
  recipe or a much smaller regularization weight.

## Granite English-r2 Run Ledger (Local Rust-Warp, May 2026)

Config: `config-granite-english-r2.yaml`, using
`ibm-granite/granite-embedding-english-r2` with the same v23/v26 recipe as the
311M multilingual run. Architecture is ModernBERT 22-layer hidden 768 GeGLU,
but with English tokenizer vocab 50,368 and RoPE theta 10,000 / 80,000.

| Run | Main config change | Checkpoint | NFCorpus NDCG@10 |
|---|---|---:|---:|
| v1 | English-r2 baseline, random projection | step_0 | **0.3590686346106829** |
| v1 | full finetune, v23/v26 recipe | step_200 | 0.3471296234218615 |
| v1 | same | step_400 | 0.34102973207707177 |
| v2-projonly | encoder frozen, projection-only | step_100 | 0.13067705225726314 |
| v3-projonly-zfix | encoder frozen, corrected XTR Z | step_50 | 0.15911275074853537 |
| v4-full-zfix | full encoder, corrected XTR Z, short warmup | step_50 | 0.12856418963337646 |
| v5-full-zfix-v1sched | full encoder, corrected XTR Z, v1 warmup/schedule | step_0 | 0.3484950399179692 |
| v5-full-zfix-v1sched | full encoder, corrected XTR Z, v1 warmup/schedule | step_50 | 0.2707780817793645 |
| v5-full-zfix-v1sched | same | step_100 | 0.12595716019256176 |
| v6-full-zfix-nokd-v1sched | v5 step_0 resume, corrected XTR Z, no KD | step_50 | 0.279277059003164 |
| v6-full-zfix-nokd-v1sched | same | step_100 | 0.13791995469223667 |
| v7-maxsim100-zfix-nokd | v5 step_0 resume, MaxSim for 100 steps then corrected XTR, no KD | step_50 | 0.34141646421136407 |
| v7-maxsim100-zfix-nokd | same, still before XTR gradients | step_100 | 0.3419035655107382 |
| v7-maxsim100-zfix-nokd | same, 50 corrected-XTR steps after handoff | step_150 | 0.10873317574270251 |
| v8-kfull-zfix-nokd | v5 step_0 resume, no KD, `k_train=65536` full-pool shortcut | step_50 | 0.3453444775293186 |
| v8-kfull-zfix-nokd | same | step_100 | 0.34676139765914116 |
| v9-k4096-zfix-nokd | v5 step_0 resume, no KD, `k_train=4096` | step_50 | **0.3494466596962226** |
| v9-k4096-zfix-nokd | same | step_100 | 0.3271258038901174 |
| v10-k1024-zfix-nokd | v5 step_0 resume, no KD, `k_train=1024` | step_50 | 0.3329474815318499 |
| v10-k1024-zfix-nokd | same | step_100 | 0.28686673781299865 |
| v11-k64-relu-zfix-nokd | v5 step_0 resume, no KD, `k_train=64`, ReLU-clamped `f_XTR` | step_50 | 0.15761380048596443 |
| v12-k8192-zfix-nokd | v5 step_0 resume, no KD, `k_train=8192` | step_50 | 0.34267918277531406 |
| v12-k8192-zfix-nokd | same | step_100 | 0.340490883944863 |
| v13-k4096-relu-zfix-nokd | v5 step_0 resume, no KD, `k_train=4096`, ReLU-clamped `f_XTR` | step_50 | 0.3470440420194771 |
| v13-k4096-relu-zfix-nokd | same | step_100 | 0.33058523383105254 |
| v14-k16384-zfix-nokd | v5 step_0 resume, no KD, `k_train=16384` | step_50 | 0.3432239610370492 |
| v14-k16384-zfix-nokd | same | step_100 | 0.3379443928677486 |
| v15-k16384-zfix-nokd-cont | resume v14 step_100, no KD, `k_train=16384`, warmup reset to 0 | step_50 | 0.3395304206510894 |
| v15-k16384-zfix-nokd-cont | same, 200 total high-k steps if counted from v14 | step_100 | 0.3437295917989028 |
| v16-bs32-k256-zfix-nokd | v5 step_0 resume, no KD, actual batch_size=32, grad_accum=1, `k_train=256` | step_50 | 0.3281001016577698 |
| v16-bs32-k256-zfix-nokd | same | step_100 | 0.2634322997659589 |
| v17-bs32-k256-projonly-zfix-nokd | v16 settings, but encoder frozen / projection-only | step_50 | 0.30055445122052543 |
| v17-bs32-k256-projonly-zfix-nokd | same | step_100 | 0.2973973309085422 |
| v18-bs32-k256-lr3e6-zfix-nokd | v16 settings, encoder unfrozen, single low `lr=3e-6` for encoder and projection | step_50 | 0.34500934543343975 |
| v18-bs32-k256-lr3e6-zfix-nokd | same | step_100 | **0.3513340692717066** |
| v19-bs32-k256-lr3e6-zfix-nokd-epoch | v18 recipe from same resume point, full-epoch schedule, save every 200; stopped manually after step_4050 | step_200 | 0.34425276124392223 |
| v19-bs32-k256-lr3e6-zfix-nokd-epoch | same | step_400 | 0.34377258129590543 |
| v19-bs32-k256-lr3e6-zfix-nokd-epoch | same | step_800 | 0.3480158020515511 |
| v19-bs32-k256-lr3e6-zfix-nokd-epoch | same | step_1600 | 0.3439881477087556 |
| v19-bs32-k256-lr3e6-zfix-nokd-epoch | same | step_2800 | 0.334983643723162 |
| v19-bs32-k256-lr3e6-zfix-nokd-epoch | same | step_3200 | 0.3312201060117117 |
| v20-bs32-k256-lr3e6-zfix-kd025 | v18 recipe plus paper-correct global XTR-KD (`kd_weight=0.25`, `kd_k_train=256`) | step_50 | 0.33536295671582567 |
| v20-bs32-k256-lr3e6-zfix-kd025 | same | step_100 | 0.34368196035347925 |
| v20-bs32-k256-lr3e6-zfix-kd025 | same | step_150 | 0.347432746317281 |
| v20-bs32-k256-lr3e6-zfix-kd025 | same | step_200 | 0.34332942451972154 |
| v20-bs32-k256-lr3e6-zfix-kd025 | same | step_250 | 0.3452534734152802 |
| v20-bs32-k256-lr3e6-zfix-kd025 | same | step_350 | 0.3472831060826292 |
| v20-bs32-k256-lr3e6-zfix-kd025 | same | step_400 | 0.34883493896584106 |
| v21-bs32-k256-lr3e6-zfix-nokd-temp1 | v18 recipe, no KD, `temperature=1.0` to match unsharpened paper CE | step_50 | 0.34680456097304174 |
| v21-bs32-k256-lr3e6-zfix-nokd-temp1 | same | step_100 | 0.3446980688909817 |
| v21-bs32-k256-lr3e6-zfix-nokd-temp1 | same | step_400 | 0.34156096598623464 |
| v22-bs128-k256-lr3e6-zfix-nokd | v18 recipe, but actual `batch_size=128`, `gradient_accumulation_steps=1`, `k_train=256` | step_50 | 0.33830931476146775 |
| v22-bs128-k256-lr3e6-zfix-nokd | same | step_100 | 0.34686554506980627 |
| v22-bs128-k256-lr3e6-zfix-nokd | same | step_400 | 0.3471974473647937 |
| v23-sigreg-from-v18 | resume best trained English checkpoint v18 step_100, add `sigreg_weight=1e-3` | step_25 | 0.34190186364649217 |
| v23-sigreg-from-v18 | same | step_50 | 0.3465355995540743 |
| v24-feather50 | resume v18 step_100, Feather all Linear weights, ramp to 50% unstructured sparsity over 100 steps | step_25 (10.6% sparse) | 0.3442047360230619 |
| v24-feather50 | same | step_50 (23.0% sparse) | 0.33454821466462 |
| v25-feather10-ramp400 | resume v18 step_100, Feather all Linear weights, ramp to 10% unstructured sparsity over 400 steps | step_100 (2.3% sparse) | 0.3456755252478917 |
| v25-feather10-ramp400 | same | step_200 (4.8% sparse) | 0.34648477586971 |
| v25-feather10-ramp400 | same | step_300 (7.3% sparse) | 0.3436502417126949 |
| v25-feather10-ramp400 | same | step_400 (9.8% sparse) | 0.3443317392424243 |

Interpretation:
- English-r2 is already stronger at `step_0` than every trained multilingual
  Granite checkpoint tested so far.
- Full encoder finetuning is destructive for English-r2 on NFCorpus, so the
  v1 run was stopped after `step_400` confirmed the decline.
- Projection-only training also collapsed the NFCorpus/Warp score by step_100
  despite rapidly lowering the MSMARCO/KD training loss. For English-r2, the
  current training objective is anti-correlated with the strong pretrained
  retrieval geometry.
- Correcting XTR's `Z` normalizer did not rescue English-r2 projection-only
  training; step_50 still collapsed to 0.159. This points beyond the Z bug:
  the MSMARCO/KD objective is overwriting a projection geometry that already
  works well for NFCorpus/Warp.
- The first corrected-Z full run used a short warmup and was not comparable to
  the earlier v1 curve. A fair rerun with the original v1 warmup/schedule still
  collapsed from a strong 0.348 step_0 baseline to 0.271 at step_50 and 0.126
  at step_100. The short schedule made the failure more abrupt, but corrected-Z
  full training is itself destructive under the current recipe.
- A no-KD ablation from the same strong v5 `step_0` checkpoint also collapsed
  under corrected-Z training: 0.279 at step_50 and 0.138 at step_100. KD may
  amplify or interact with the failure, but it is not required for collapse.
- Delaying corrected-XTR with a 100-step MaxSim warmup preserves the checkpoint
  through the MaxSim phase (0.341 at step_50, 0.342 at step_100). The first 50
  corrected-XTR steps after the handoff collapse to 0.109, and the training loss
  jumps from ~1.14 to 8.09 immediately after the switch. The failure is tied to
  the corrected-XTR objective itself, not to cold-starting from random top-k
  alignments.
- Full-pool XTR (`k_train=65536`) does not collapse over 100 steps and behaves
  like MaxSim, scoring 0.345 at step_50 and 0.347 at step_100. This rules out
  the corrected-Z implementation as the direct collapse mechanism; the sparse
  top-k masking is the destructive part.
- Sparse high-k has a sharp stability gradient. `k_train=4096` gives the best
  short-run checkpoint (0.349 at step_50) but drifts to 0.327 by step_100.
  `k_train=1024` drops harder (0.333 -> 0.287), while `k_train=8192` and
  `k_train=16384` stay in the 0.338-0.343 band through step_100.
- ReLU-clamping final `f_XTR` is not a sufficient fix. At `k_train=64` it still
  collapses by step_50 (0.158). At `k_train=4096` it slightly improves the
  step_100 score versus unclamped 4096 (0.331 vs 0.327), but it also lowers the
  step_50 peak and does not prevent drift.
- A continuation from `k_train=16384` step_100 with a fresh optimizer and no
  warmup reached 0.344 at its own step_100. Lower training loss did not recover
  the original English-r2 baseline, so this is still fitting away from the
  pretrained NFCorpus/Warp geometry.
- Reducing the actual XTR token pool by lowering batch size (`batch_size=32`,
  `gradient_accumulation_steps=1`, `k_train=256`) softened but did not remove
  drift: 0.328 at step_50 and 0.263 at step_100. The smaller XTR pool avoids
  the catastrophic k64 failure mode, but full-encoder MSMARCO training still
  moves away from the pretrained NFCorpus/Warp geometry.
- Freezing the encoder with the same `batch_size=32` / `k_train=256` setup made
  the result much more stable but lower overall: 0.301 at step_50 and 0.297 at
  step_100. This suggests encoder movement is responsible for the late v16
  drop, but projection-only XTR still does not recover the strong pretrained
  English-r2 geometry.
- Lowering the unfrozen full-model LR to a single low `lr=3e-6`
  (`lr=3e-6` for both encoder and projection) prevented the bs32/k256 collapse
  through step_100 and produced the best corrected-Z English-r2 checkpoint so
  far: 0.345 at step_50 and 0.351 at step_100. This points to LR/projection-LR
  scale as the immediate collapse trigger, not unfrozen encoder training by
  itself.
- Extending the same low-LR recipe toward a full epoch did not keep improving.
  v19 stayed healthy through step_800 (0.348), but drifted down by step_2800
  (0.335) and step_3200 (0.331), so the run was stopped at step_4050. The
  practical window for this no-KD recipe appears early; more steps overfit or
  move away from NFCorpus/Warp geometry even at low LR.
- Paper-correct XTR-KD is stable but has not beaten the no-KD low-LR run. v20
  applies XTR Eq. 2 top-k over the full KD mini-batch document-token pool, then
  gathers each query's teacher-scored candidates for KL. With `kd_weight=0.25`,
  it recovers from 0.335 at step_50 to 0.349 at step_400, but remains below the
  no-KD v18 step_100 score of 0.351.
- Removing the contrastive temperature to more literally match the paper's
  unsharpened CE objective did not help. v21's training loss stayed near 3.96
  and NFCorpus/Warp drifted from 0.347 at step_50 to 0.342 at step_400. For
  our L2-normalized Granite setup, temperature sharpening is still useful even
  if the original paper does not call out an explicit temperature.
- Moving toward the paper's larger actual batch regime did not recover the
  missing performance. v22 used actual `batch_size=128` with no gradient
  accumulation and `k_train=256`; it recovered from 0.338 at step_50 to 0.347
  by step_400, but stayed below the smaller-batch v18 peak. The larger
  contrastive pool is not the direct fix for Granite.
- A short SIGREG continuation from the best trained English checkpoint did not
  help. v23 resumed v18 step_100 (0.3513) and added weak pre-normalization
  SIGREG (`sigreg_weight=1e-3`); it scored 0.342 at step_25 and 0.347 at
  step_50. The local PQ bucket distribution was effectively unchanged versus
  the v18 start point: v18 step_100 `p95=208 p99=399 max=2205`, v23 step_50
  `p95=208 p99=404 max=2235`.
- Feather unstructured sparsity works mechanically but costs quality quickly.
  v24 wrapped all `nn.Linear` weights with the Feather transform, used
  threshold sampling for the 149M-weight model, and baked sparse effective
  weights into checkpoints. Around 10.6% baked sparsity scored 0.344, and
  around 23.0% scored 0.335. The step_50 bucket distribution did not expose
  a compensating PQ win (`p95=203 p99=373 max=2374`), and the training loss
  rose sharply past 30% sparsity, so step_75/step_100 were not evaluated.
- Slowing Feather down helped stability but did not preserve the dense score.
  v25 ramped to 10% sparsity over 400 steps; best was step_200 at 4.8% sparsity
  with 0.3465, still below the dense v18 checkpoint at 0.3513. Bucket max
  improved at intermediate sparsity (step_100 max=2059, step_200 max=1813,
  step_300 max=1612) but NDCG did not improve, and final step_400 drifted to
  0.3443 with bucket stats `p95=204 p99=400 max=1971`.
- The multilingual Granite gains should not be interpreted as clean evidence of
  XTR-specific optimality. They are confounded by English/MSMARCO adaptation and
  projection fitting; English-r2 shows that the base English encoder is already
  substantially better than the trained multilingual checkpoints.

Key findings:
- **Full model training destroys the encoder.** Catastrophic forgetting on
  NFCorpus when finetuning on English MSMARCO. Scores degrade continuously.
- **Frozen encoder + projection only is the right approach.** Preserves the
  encoder's quality while allowing the projection to adapt.
- **Temperature matters.** With L2-normalized embeddings, temp=1.0 gives
  near-uniform softmax (loss ~5.5 vs random ~6.2). temp=0.05 gives meaningful
  gradients (loss drops to ~1.5). But lower loss doesn't always mean better
  Warp scores.
- **Warp scores plateau around 0.25.** Despite significant improvements in
  Python MaxSim and XTR scores, Warp centroid retrieval improves only modestly
  (0.250 → 0.255). The gap appears structural, not addressable by training
  the projection alone.

## UltraBERT-150M Granite-v18 Replica (October 20, 2026)

Recovered the successful Granite English R2 encoder/projection policy with
`pickbrain`, including the June 24 explanation of v18's fixed step-zero
projection and the August 31 summary of the later 96d/gate curriculum.
This experiment reproduces the v18 encoder/projection stage using
[`dmis-lab/UltraBERT-150M`](https://huggingface.co/dmis-lab/UltraBERT-150M),
revision `545ac9d9c7aece444f014ada5a1e9818132b4d35`.
UltraBERT is an MLM base encoder; Granite English R2 already has embedding
training, so this measures recipe transfer between different starting points.

Recipe: full encoder plus a fresh random 128d linear projection, seed 42,
MSMARCO MiniLM-L6-v3 English triplets, batch 32, query/document lengths 32/256,
corrected-Z XTR with `k_train=256`, temperature 0.05, shared LR `3e-6`, AdamW,
weight decay 0.01, gradient clipping 1.0, BF16, and gradient checkpointing.
No KD or encoder freezing. The 200-step warmup is capped at the 100-step
training horizon, exactly as in v18; the entire trial is an LR ramp.
Granite's saved random projection cannot be resumed into the different base
model as a complete checkpoint, so UltraBERT uses its own seeded initialization.

Evaluation uses **exact all-token MaxSim**, all 3,633 NFCorpus documents and
323 test queries, with the same lengths and 128d vectors for both models.
These scores must not be compared directly with the Rust-Warp table above.

| Model/checkpoint | Exact MaxSim NDCG@10 | Recall@100 |
|---|---:|---:|
| UltraBERT step_0 | 0.1471145118 | 0.1646767937 |
| UltraBERT step_50 | **0.1818494351** | **0.2067770006** |
| UltraBERT step_100 | 0.1404513173 | 0.1865636801 |
| Granite English R2 v18 step_100, re-evaluated | **0.3685125599** | **0.3249455198** |

The best UltraBERT checkpoint is step_50: +0.03473 absolute NDCG@10 (+23.6%)
over its random-projection baseline, but 0.18666 below Granite under the same
scorer. Step_100 is below UltraBERT's own baseline. Training remained finite;
cumulative mean XTR loss reached 3.7291 at step 60, then rose to 3.8857 by
step 100. This single-seed, short-run result rejects direct transfer of the
v18 policy as a competitive UltraBERT recipe; it does not establish UltraBERT's
quality after a different retrieval-training schedule.

All three checkpoints are preserved on `gpu` under
`/home/jhansen/xtrbert-runs/ultrabert-150m-v1-20261020/checkpoints-ultrabert-150m-xtr-v1-granite-v18/`.
The job completed and released the GPU. It used Python 3.11.7, PyTorch
2.14.0+cu130, Transformers 5.5.3, and an NVIDIA L4; the existing model wrapper
worked without architecture changes or a package upgrade.

Reproduction: use [the experiment config](config-ultrabert-150m-xtr-v1-granite-v18.yaml)
through `python run_ultrabert_xtr.py` in a checkout with `beir/nfcorpus`
available. [The runner](run_ultrabert_xtr.py) pins the model revision, seeds
training, evaluates steps 0/50/100, and records the best checkpoint.
Pass `--reference-checkpoint PATH` to re-evaluate Granite with the same scorer;
pass `--evaluate-only` to reuse an existing completed run.
The manifest, metrics, losses, and complete execution log are retained in
[`experiments/ultrabert-150m-v1/`](experiments/ultrabert-150m-v1/).

## Training Loss (XTR vs MaxSim)

**MaxSim (ColBERT)**: For each query token, take max similarity over all tokens
in each document. Average over query tokens. Contrastive cross-entropy over docs.

**XTR (Eq. 2)**: For each query token, find its top-k matches across ALL doc
tokens in the batch. Scatter these into per-document max similarities.
Normalize by Z = number of query tokens that matched ≥1 token from each document
(not total query tokens). Unmatched (query_token, doc) pairs contribute 0.

## Training Setup

Config: `config-colmodernvbert-xtr-projonly.yaml`

```yaml
dataset: sentence-transformers/msmarco-msmarco-MiniLM-L6-v3  # model-mined hard negs
batch_size: 256       # limited by A10G 24GB
k_train: 128
temperature: 0.05
lr: 5.0e-4           # higher LR for projection-only
freeze_encoder: true
fp16: true
```

The `build_optimizer` function in train.py supports separate `encoder_lr` and
head `lr` for differential learning rates if needed.

## Hardware

- **A10G (24GB)**: batch_size=256 works, 320 OOM's on batches with long documents.
  The sim matrix `(B, query_len, B × doc_len)` is the memory bottleneck and scales
  quadratically with batch size.
- **Local Mac (MPS)**: Smoke tests only (batch_size=4-8, max_samples=64).
- **Harrier (270M params)**: ~80% more memory than ColModernVBert (150M).

## Evaluation

`eval_full_pipeline.sh [checkpoint_dir]` runs the full loop:
export → Warp build → embed/index/query → score → Python brute-force.

`eval_nfcorpus.py [--checkpoint dir]` runs Python-only MaxSim and XTR scoring.

`plot_loss.py <checkpoint_dir> [window]` visualizes loss, derivative, and LR.

## Critical Bug Fix: Rust Sliding Window Attention

ModernBERT uses sliding window local attention (window=128) on most layers,
with global attention only every 3rd layer (0, 3, 6, ..., 21). This was
missing from the Rust implementation, causing all layers to use global attention.
Token-level cosine similarity was 0.947 before the fix, 0.9999 after.

**All Warp evaluation numbers from before this fix are invalid.**

## Critical Bug Fix: XTR Z-Normalizer

XTR Eq. 2 defines `Z` as the number of query tokens that retrieved at least one
token from the scored document: `|{i | exists j, A_ij > 0}|`. Our training code
incorrectly used the number of query tokens with a positive retained score:
`max_sim > 0`.

That is not equivalent. If a query token retrieved a document token with a
non-positive retained score, the paper still counts that query token in `Z`;
the old code did not. This could inflate sparse positive hits and make the loss
less faithful to XTR's "retrieved tokens must be useful" pressure.

Fixed in `train.py`: both contrastive XTR scoring and KD-XTR scoring now compute
`Z` from the top-k selection counts, masked by valid query tokens. Results before
this fix are useful operationally, but should not be treated as paper-faithful
XTR training results.

## Lessons Learned

1. **The infinite training loop with finite cosine schedule was a major bug.**
   After the schedule horizon, HF's scheduler wraps around and LR climbs again,
   destabilizing converged models. Fixed: stop stepping the scheduler after
   total_steps. Now training stops cleanly at the schedule horizon.

2. **Recipe drift caused misleading results.** Configs labeled "paper" used
   BM25 negatives, temperature=0.05, and k_train=64 — none of which match
   the actual paper. Always verify configs against the paper.

3. **The random CE baseline depends on effective batch size.** With negatives
   appended, softmax has 2×batch_size docs: random CE = ln(512) ≈ 6.24,
   not ln(256) ≈ 5.55.

4. **Gradient accumulation is not a larger contrastive/XTR batch.**
   Accumulation averages gradients from independent microbatch softmaxes; it
   does not enlarge the in-batch negative pool or the XTR top-k token pool.
   Runs with `batch_size=128, gradient_accumulation_steps=2` should be read as
   two accumulated batch-128 objectives, not a true batch-256 contrastive loss.

5. **Python eval k_eval is hardcoded.** eval_nfcorpus.py uses k_eval=100,
   which doesn't match Warp's retrieval budget. Useful for relative comparison
   but not paper-faithful.

## Open Questions

1. **Why does the 0.34→0.25 gap persist?** Training improves Python scores
   but barely moves Warp. The centroid retrieval gap may be structural —
   related to k-means quality, Q4 quantization, or n_probe budget rather
   than embedding quality.

2. **MLP projection?** A nonlinear projection (768→384→128) might have
   enough capacity to optimize for both MaxSim and clustering geometry
   simultaneously.

3. **Warp-side tuning?** Instead of changing embeddings, improve centroid
   retrieval: more k-means iterations, larger n_probe, better quantization.

4. **Differential LR for partial unfreezing?** Freeze most layers, unfreeze
   last 2-3 encoder layers with a small LR (1e-6) + projection with larger LR.
   May get benefits of encoder adaptation without catastrophic forgetting.

5. **Evaluate on MSMARCO dev.** We only evaluate on NFCorpus. The training
   might be improving MSMARCO scores while being neutral/negative on NFCorpus
   (domain shift).
