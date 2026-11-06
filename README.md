# Awesome Circuit-Preserving Compression [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated, comprehensive reading list at the intersection of **LLM compression** (pruning, quantization) and **mechanistic interpretability** (circuits, features, capability localization), oriented toward **edge deployment**.

This list was assembled while researching a specific topic — *selective circuit preservation during compression*: identify the circuits that implement a chosen set of capabilities, protect them while the rest of the model is compressed, and verify with circuit-faithfulness tests that the capabilities survived. The categories below follow that research arc: from classic compression, to the evidence that compression is capability-blind, to the interpretability toolkit that can localize capabilities, to the small but growing body of work that connects the two, to the hardware that has to run the result.

Corrections and additions welcome — open an issue or PR.

## Contents

- [Foundational Pruning & Compression](#foundational-pruning--compression)
- [LLM Pruning](#llm-pruning)
- [Quantization](#quantization)
- [Low-Rank Factorization](#low-rank-factorization)
- [Knowledge Distillation](#knowledge-distillation)
- [KV-Cache / Activation Compression](#kv-cache--activation-compression)
- [Conditional Computation](#conditional-computation)
- [Joint & Unified Compression](#joint--unified-compression)
- [Compression Breaks Capabilities](#compression-breaks-capabilities)
- [Interpretability: Superposition, Features & SAEs](#interpretability-superposition-features--saes)
- [Interpretability: Circuits](#interpretability-circuits)
- [Interpretability: Knowledge & Function Localization](#interpretability-knowledge--function-localization)
- [Interpretability: Automated Circuit Discovery & Attribution](#interpretability-automated-circuit-discovery--attribution)
- [Interpretability: Verification & Faithfulness](#interpretability-verification--faithfulness)
- [Capabilities as Manipulable Objects](#capabilities-as-manipulable-objects)
- [Circuit-Aware & Capability-Aware Compression (closest prior art)](#circuit-aware--capability-aware-compression-closest-prior-art)
- [Edge Deployment & Hardware Realization](#edge-deployment--hardware-realization)
- [Theoretical Foundations](#theoretical-foundations)

---

## Foundational Pruning & Compression

- **Optimal Brain Damage.** LeCun, Denker, Solla. NeurIPS 1990. — Second-derivative saliency for weight removal; the origin of saliency-based pruning.
- **Second Order Derivatives for Network Pruning: Optimal Brain Surgeon.** Hassibi, Stork. NeurIPS 1993. — Curvature-aware pruning that also updates the surviving weights.
- **Learning both Weights and Connections for Efficient Neural Networks.** Han, Pool, Tran, Dally. 2015. [arXiv:1506.02626](https://arxiv.org/abs/1506.02626) — Magnitude pruning + retraining, the modern pruning baseline.
- **Deep Compression.** Han, Mao, Dally. ICLR 2016. [arXiv:1510.00149](https://arxiv.org/abs/1510.00149) — Pruning + trained quantization + Huffman coding in one pipeline.
- **To prune, or not to prune: exploring the efficacy of pruning for model compression.** Zhu, Gupta. 2017. [arXiv:1710.01878](https://arxiv.org/abs/1710.01878) — The gradual-magnitude-pruning schedule most later one-shot/iterative pruning methods are compared against.
- **The Lottery Ticket Hypothesis.** Frankle, Carbin. ICLR 2019. [arXiv:1803.03635](https://arxiv.org/abs/1803.03635) — Sparse subnetworks that train to full accuracy from the original init.
- **Rethinking the Value of Network Pruning.** Liu, Sun, Zhou, Huang, Darrell. ICLR 2019. [arXiv:1810.05270](https://arxiv.org/abs/1810.05270) — Argues the pruned *architecture*, not the inherited weights, is what carries over — a foundational counterpoint to "important weight" framings.
- **SNIP: Single-shot Network Pruning based on Connection Sensitivity.** Lee, Ajanthan, Torr. ICLR 2019. [arXiv:1810.02340](https://arxiv.org/abs/1810.02340) — Prunes before training using a single gradient-based saliency pass.
- **GraSP: Picking Winning Tickets Before Training by Preserving Gradient Flow.** Wang, Zhang, Grosse. ICLR 2020. [arXiv:2002.07376](https://arxiv.org/abs/2002.07376) — Pruning criterion that protects gradient flow rather than loss.
- **Movement Pruning: Adaptive Sparsity by Fine-Tuning.** Sanh, Wolf, Rush. NeurIPS 2020. [arXiv:2005.07683](https://arxiv.org/abs/2005.07683) — First-order pruning driven by weight movement during fine-tuning rather than magnitude, the transfer-learning-era precursor to LLM pruning criteria.
- **SynFlow: Pruning Neural Networks without any Data.** Tanaka, Kunin, Yamins, Ganguli. NeurIPS 2020. [arXiv:2006.05467](https://arxiv.org/abs/2006.05467) — Data-free pruning via synaptic flow conservation.
- **Optimal Brain Apoptosis.** Sun, Fang, Wang, et al. 2025. [arXiv:2502.17941](https://arxiv.org/abs/2502.17941) — Second-order pruning criterion scaled to modern LLMs.

## LLM Pruning

- **SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot.** Frantar, Alistarh. ICML 2023. [arXiv:2301.00774](https://arxiv.org/abs/2301.00774) — One-shot ≥50% unstructured sparsity via layer-wise second-order reconstruction.
- **Wanda: A Simple and Effective Pruning Approach for LLMs.** Sun, Liu, Bair, Kolter. 2023. [arXiv:2306.11695](https://arxiv.org/abs/2306.11695) — Prunes by weight magnitude × input activation norm, no retraining or Hessians.
- **LLM-Pruner: On the Structural Pruning of Large Language Models.** Ma, Fang, Wang. 2023. [arXiv:2305.11627](https://arxiv.org/abs/2305.11627) — Removes coupled structures (not just individual weights) with gradient-based grouping.
- **Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning.** Xia, Gao, Zeng, Chen. 2023. [arXiv:2310.06694](https://arxiv.org/abs/2310.06694) — Structured pruning of a larger model plus targeted continued pretraining beats training a same-size model from scratch at 3% of the compute.
- **OWL: Outlier Weighed Layerwise Sparsity.** Yin et al. ICML 2024. [arXiv:2310.05175](https://arxiv.org/abs/2310.05175) — Allocates per-layer sparsity ratios by outlier-weight density.
- **AMP: Attention Heads and MLP Pruning for Efficient LLMs.** Mugnaini et al. 2025. [arXiv:2504.21174](https://arxiv.org/abs/2504.21174) — Structured removal of whole attention heads and MLP units.
- **Entropy Meets Importance: A Unified Head Importance-Entropy Score.** Choi, Son, Kim, Kim. 2025. [arXiv:2510.13832](https://arxiv.org/abs/2510.13832) — Combines importance and entropy signals for stable head pruning.
- **From LLMs to LRMs: Rethinking Pruning for Reasoning-Centric Models.** Ding, Zhao, Ye, Chen, Shen. 2026. [arXiv:2601.18091](https://arxiv.org/abs/2601.18091) — Standard pruning criteria under-protect multi-step reasoning ability.
- **Diet Your LLM: Dimension-wise Global Pruning via Merging Task-specific Importance.** Hong, Kim. 2026. [arXiv:2603.23985](https://arxiv.org/abs/2603.23985) — Merges per-task importance scores for a single global pruning mask.
- **Accelerated Sparse Neural Training: Provable N:M Transposable Masks.** Hubara, Chmiel, Island, Banner, Naor, Soudry. 2021. [arXiv:2102.08124](https://arxiv.org/abs/2102.08124) — The N:M structured-sparsity formulation hardware accelerators rely on.

## Quantization

- **Qualcomm Dragonwing QCS8550.** Product page. [qualcomm.com](https://www.qualcomm.com/internet-of-things/products/q8-series/qcs8550) — INT8 NPU / FP16 GPU edge SoC used as the representative deployment target.
- **k-quants.** Kawrakow. llama.cpp PR #1684, 2023. [github.com/ggml-org/llama.cpp/pull/1684](https://github.com/ggml-org/llama.cpp/pull/1684) — The GGUF block/super-block quantization scheme used by llama.cpp.
- **LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.** Dettmers, Lewis, Belkada, Zettlemoyer. NeurIPS 2022. [arXiv:2208.07339](https://arxiv.org/abs/2208.07339) — Mixed-precision decomposition that isolates outlier feature dimensions at 16-bit; the foundational LLM weight+activation quantization result.
- **GPTQ: Accurate Post-Training Quantization for GPT.** Frantar, Ashkboos, Hoefler, Alistarh. 2022. [arXiv:2210.17323](https://arxiv.org/abs/2210.17323) — One-shot 3–4-bit weight quantization without retraining.
- **QLoRA: Efficient Finetuning of Quantized LLMs.** Dettmers, Pagnoni, Holtzman, Zettlemoyer. NeurIPS 2023. [arXiv:2305.14314](https://arxiv.org/abs/2305.14314) — 4-bit NormalFloat quantization plus double quantization and paged optimizers, enabling 65B-parameter fine-tuning on one 48GB GPU.
- **AWQ: Activation-aware Weight Quantization.** Lin, Tang, Tang, et al. 2023. [arXiv:2306.00978](https://arxiv.org/abs/2306.00978) — Protects the ~1% most salient weights (by activation statistics) via per-channel scaling, not mixed-precision storage.
- **SmoothQuant: Accurate and Efficient PTQ for LLMs.** Xiao, Lin, Seznec, Wu, Demouth, Han. 2022. [arXiv:2211.10438](https://arxiv.org/abs/2211.10438) — Migrates activation quantization difficulty into the weights via a smoothing factor.
- **HAWQ: Hessian AWare Quantization.** Dong, Yao, Gholami, Mahoney, Keutzer. ICCV 2019. [arXiv:1905.03696](https://arxiv.org/abs/1905.03696) — Second-order sensitivity to set per-layer mixed precision.
- **SqueezeLLM: Dense-and-Sparse Quantization.** Kim, Hooper, Gholami, et al. 2023. [arXiv:2306.07629](https://arxiv.org/abs/2306.07629) — Splits weights into a dense low-bit component and a sparse high-precision outlier component.
- **SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression.** Dettmers, Svirschevski, et al. 2023. [arXiv:2306.03078](https://arxiv.org/abs/2306.03078) — Isolates outlier weights at higher precision alongside a low-bit bulk.
- **APTQ: Attention-aware Post-Training Mixed-Precision Quantization.** Guan, Huang, Su, Huang, Wong, Yu. 2024. [arXiv:2402.14866](https://arxiv.org/abs/2402.14866) — Allocates bit-width per attention block by sensitivity.
- **BitNet b1.58: The Era of 1-bit LLMs.** Ma, Wang, et al. 2024. [arXiv:2402.17764](https://arxiv.org/abs/2402.17764) — Ternary {-1,0,1} weights trained from scratch.
- **ParetoQ: Improving Scaling Laws in Extremely Low-bit LLM Quantization.** Liu, Zhao, Huang, et al. 2025. [arXiv:2502.02631](https://arxiv.org/abs/2502.02631) — Unified training recipe spanning 1–4 bit quantization-aware training.
- **A White Paper on Neural Network Quantization.** Nagel, Fournarakis, Amjad, Bondarenko, van Baalen, Blankevoort. 2021. [arXiv:2106.08295](https://arxiv.org/abs/2106.08295) — Practitioner-oriented survey of PTQ and QAT.
- **FPTQuant: Function-Preserving Transforms for LLM Quantization.** van Breugel, Bondarenko, Whatmough, Nagel. ICML 2026. [arXiv:2506.04985](https://arxiv.org/abs/2506.04985) — Lightweight transforms that reshape activation distributions for quantization while providably preserving the function computed.
- **Leech Lattice Vector Quantization for Efficient LLM Compression.** van der Ouderaa, van Baalen, Whatmough, Nagel. 2026. [arXiv:2603.11021](https://arxiv.org/abs/2603.11021) — Encodes weights in blocks using the 24-dimensional Leech lattice's optimal packing, outperforming QuIP#/QTIP-style vector quantization.

## Low-Rank Factorization

- **LASER: Layer-Selective Rank Reduction.** Sharma, Ash, Misra. 2023. [arXiv:2312.13558](https://arxiv.org/abs/2312.13558) — Low-rank truncation of specific weight matrices *improves* reasoning accuracy.
- **ASVD: Activation-aware Singular Value Decomposition.** Yuan et al. 2023. [arXiv:2312.05821](https://arxiv.org/abs/2312.05821) — Weights SVD by activation magnitude before truncation.
- **SVD-LLM: Truncation-aware Singular Value Decomposition.** Wang et al. 2024. [arXiv:2403.07378](https://arxiv.org/abs/2403.07378) — Closed-form loss compensation for SVD truncation error.
- **AdaSVD: Adaptive Singular Value Decomposition.** Li, Xia, Zhang, et al. 2025. [arXiv:2502.01403](https://arxiv.org/abs/2502.01403) — Per-layer adaptive rank allocation for SVD compression.
- **UniRank: Unified Rank Allocation for Low-Rank LLM Compression.** Han, Hu, Ma, Zhang, Shen. 2026. [arXiv:2606.21847](https://arxiv.org/abs/2606.21847) — Scores components by local singular-energy ratio and global functional importance to set per-layer rank, beating uniform allocation.

## Knowledge Distillation

- **Distilling the Knowledge in a Neural Network.** Hinton, Vinyals, Dean. 2015. [arXiv:1503.02531](https://arxiv.org/abs/1503.02531) — The original soft-label distillation formulation.
- **DistilBERT.** Sanh, Debut, Chaumond, Wolf. 2019. [arXiv:1910.01108](https://arxiv.org/abs/1910.01108) — 40% smaller BERT retaining 97% of language-understanding performance.
- **TinyBERT: Distilling BERT for Natural Language Understanding.** Jiao, Yin, Shang, et al. EMNLP 2020. [arXiv:1909.10351](https://arxiv.org/abs/1909.10351) — Two-stage (pretraining + task-specific) transformer distillation across embeddings, attention, and hidden states, not just output logits.
- **MiniLLM: Knowledge Distillation of Large Language Models.** Gu, Dong, Wei, Huang. 2023. [arXiv:2306.08543](https://arxiv.org/abs/2306.08543) — Reverse-KLD objective tailored to generative LLM distillation.
- **Distilling Step-by-Step!** Hsieh et al. 2023. [arXiv:2305.02301](https://arxiv.org/abs/2305.02301) — Distills rationales, not just labels, into a much smaller student.
- **RLAD: Reinforcement-aware Knowledge Distillation for LLM Reasoning.** Zhang, Jiang, Shen, et al. 2026. [arXiv:2602.22495](https://arxiv.org/abs/2602.22495) — Replaces the KL-regularized distillation objective with a PPO/GRPO-style likelihood-ratio term to better preserve reasoning capability in the student.

## KV-Cache / Activation Compression

- **Efficient Streaming Language Models with Attention Sinks (StreamingLLM).** Xiao, Tian, Chen, Han, Lewis. ICLR 2024. [arXiv:2309.17453](https://arxiv.org/abs/2309.17453) — Identifies attention-sink tokens that must be kept for stable long-context generation, the foundational result behind most KV-cache eviction policies.
- **KIVI: A Tuning-Free Asymmetric 2-bit Quantization for KV Cache.** Liu et al. 2024. [arXiv:2402.02750](https://arxiv.org/abs/2402.02750) — Per-channel key / per-token value quantization for the KV cache.
- **KVQuant: Towards 10M Context Length LLM Inference.** Hooper et al. 2024. [arXiv:2401.18079](https://arxiv.org/abs/2401.18079) — Sub-4-bit KV-cache quantization for very long contexts.
- **H2O: Heavy-Hitter Oracle for Efficient Generative Inference.** Zhang et al. NeurIPS 2023. [arXiv:2306.14048](https://arxiv.org/abs/2306.14048) — Evicts KV entries by cumulative attention-score "heavy hitters".
- **SnapKV: LLM Knows What You Are Looking For Before Generation.** Li et al. NeurIPS 2024. [arXiv:2404.14469](https://arxiv.org/abs/2404.14469) — Compresses the KV cache using attention patterns from the prompt itself.
- **KV-CoRE: Benchmarking Data-Dependent Low-Rank Compressibility of KV-Caches in LLMs.** Chen, Wang, Qin, et al. 2026. [arXiv:2602.05929](https://arxiv.org/abs/2602.05929) — SVD-based framework showing KV-cache compressibility varies systematically with architecture and data rather than being a fixed cache property.

## Conditional Computation

- **Outrageously Large Neural Networks: Sparsely-Gated Mixture-of-Experts.** Shazeer, Mirhoseini, Maziarz, et al. 2017. [arXiv:1701.06538](https://arxiv.org/abs/1701.06538) — The MoE layer that routes each token to a sparse expert subset.
- **Mixture-of-Depths.** Raposo et al. 2024. [arXiv:2404.02258](https://arxiv.org/abs/2404.02258) — Per-token dynamic allocation of compute across depth.
- **Fast Inference from Transformers via Speculative Decoding.** Leviathan, Kalman, Matias. 2022. [arXiv:2211.17192](https://arxiv.org/abs/2211.17192) — Draft-then-verify decoding for lossless speedup.

## Joint & Unified Compression

- **EDGE-LLM: Layerwise Unified Compression and Adaptive Layer Tuning.** Yu et al. DAC 2024. [arXiv:2406.15758](https://arxiv.org/abs/2406.15758) — Combines compression with adaptive fine-tuning for on-device adaptation.
- **Towards Optimal Compression: Joint Pruning and Quantization.** Zandonati, Bucagu, Pol, Pierini, Sirkin, Kopetz. 2023. [arXiv:2302.07612](https://arxiv.org/abs/2302.07612) — Formulates prune+quantize as one joint optimization rather than sequential steps.
- **BESA: Blockwise Parameter-Efficient Sparsity Allocation.** Xu, Shao, Chen, et al. 2024. [arXiv:2402.16880](https://arxiv.org/abs/2402.16880) — Learns per-block sparsity ratios end-to-end.
- **Optimizing LLMs for Resource-Constrained Environments: A Survey.** Girija, Kapoor, Arora, Pradhan, Raj, Shetgaonkar. 2025. [arXiv:2505.02309](https://arxiv.org/abs/2505.02309) — Survey spanning pruning, quantization, distillation, and their combination.
- **Prune-then-Quantize or Quantize-then-Prune?** Kim, Choi, Yang, Kim, Song, Kang. ICLR 2026. [arXiv:2603.18426](https://arxiv.org/abs/2603.18426) — Shows compression *order* materially changes joint prune+quantize outcomes.
- **A Systematic Study of Compression Ordering for Large Language Models.** 2025. [arXiv:2511.19495](https://arxiv.org/abs/2511.19495) — Earlier/companion systematic study of ordering effects across compression operators.
- **UniComp: A Unified Evaluation of LLM Compression via Pruning, Quantization and Distillation.** 2026. [arXiv:2602.09130](https://arxiv.org/abs/2602.09130) — 6 methods × 40 datasets; finds factual recall survives compression far better than multi-step reasoning.
- **GETA: Automatic Joint Structured Pruning and Quantization.** Qu, Aponte, Banbury, et al. 2025. [arXiv:2502.16638](https://arxiv.org/abs/2502.16638) — A quantization-aware dependency graph plus a constrained-gradient method that learns pruning and quantization jointly rather than sequentially.

## Compression Breaks Capabilities

*The evidence motivating why capability-blind compression is a problem worth solving.*

- **Compressing LLMs: The Truth is Rarely Pure and Never Simple.** Jaiswal, Gan, Du, Zhang, Wang, Yang. 2023. [arXiv:2310.01382](https://arxiv.org/abs/2310.01382) — Knowledge-intensive tasks collapse under pruning even when perplexity looks fine.
- **Through a Compressed Lens: Quantization's Impact on Factual Knowledge Recall.** Wang, Wang, Feldhus, et al. 2025. [arXiv:2505.13963](https://arxiv.org/abs/2505.13963) — Factual-recall damage from quantization is amplified in smaller models of a family.
- **ACBench: Can Compressed LLMs Truly Act?** Dong et al. ICML 2025. [arXiv:2505.19433](https://arxiv.org/abs/2505.19433) — Agentic tool-use capability degrades faster under compression than standard benchmarks suggest.
- **The Lottery LLM Hypothesis: Rethinking What Abilities Compression Should Preserve.** Tang, Liu, Wang, Dong, He, Chu, Li. 2025. [arXiv:2502.17535](https://arxiv.org/abs/2502.17535) — Argues for capability-targeted, not uniform, compression objectives.
- **Investigating the Impact of Quantization on Safety and Reliability.** Kharinaev, Moskvoretskii, Shvetsov, et al. 2025. [arXiv:2502.15799](https://arxiv.org/abs/2502.15799) — Quantization measurably shifts safety-refusal behavior.
- **How Does Quantization Affect Multilingual LLMs?** Marchisio et al. 2024. [arXiv:2407.03211](https://arxiv.org/abs/2407.03211) — Non-English capability degrades disproportionately under quantization.
- **Interpreting the Effects of Quantization on LLMs.** Singh, Sajjad. 2025. [arXiv:2508.16785](https://arxiv.org/abs/2508.16785) — Applies interpretability probes directly to quantized models; finds no drastic representational shift at 4-/8-bit, a useful counterweight to the failure cases elsewhere in this section.
- **Emergent Abilities of Large Language Models.** Wei et al. 2022. [arXiv:2206.07682](https://arxiv.org/abs/2206.07682) — Some capabilities appear abruptly with scale — a caution for "average error" compression metrics.
- **Are Emergent Abilities of Large Language Models a Mirage?** Schaeffer, Miranda, Koyejo. NeurIPS 2023. [arXiv:2304.15004](https://arxiv.org/abs/2304.15004) — Apparent emergence can be a metric artifact, not a real phase transition.
- **Preserving LLM Capabilities through Calibration Data Curation.** 2025. [arXiv:2510.10618](https://arxiv.org/abs/2510.10618) — Calibration-set composition alone materially changes which capabilities a compressed model retains.
- **Does Compression Preserve Uncertainty?** 2026. [arXiv:2606.01850](https://arxiv.org/abs/2606.01850) — Conformal-prediction benchmark shows compression can decouple accuracy from calibrated uncertainty.
- **Weight Pruning Amplifies Bias: A Multi-Method Study for Edge AI.** 2026. [arXiv:2605.08137](https://arxiv.org/abs/2605.08137) — Pruning methods increase demographic bias disproportionately at edge-relevant sparsity levels.
- **When Reasoning Meets Compression.** 2025. [arXiv:2504.02010](https://arxiv.org/abs/2504.02010) — Large reasoning models lose chain-of-thought quality under compression in ways accuracy metrics under-report.
- **Perplexity Can Miss SAE Feature Damage Under Quantization.** 2026. [arXiv:2606.03002](https://arxiv.org/abs/2606.03002) — Sparse-autoencoder features are damaged by quantization even when perplexity is unchanged — direct evidence for why a mechanistic verification step is needed.
- **How Pruning Reshapes Features: SAE Analysis of Weight-Pruned Language Models.** 2026. [arXiv:2603.25325](https://arxiv.org/abs/2603.25325) — Pruning performs *implicit* feature selection: rare/specialized SAE features survive at far higher rates than generic high-frequency ones.
- **Super Weights in LLMs and the Failure of Selective Training.** Subramanian, Akinfaderin, Sehwag. 2026. [arXiv:2607.08733](https://arxiv.org/abs/2607.08733) — Cautionary counterpoint for this whole research direction: a parameter identified as individually critical (a "super weight") often cannot be safely isolated and trained/protected on its own — importance for the forward pass doesn't imply safe isolability.

## Interpretability: Superposition, Features & SAEs

- **Toy Models of Superposition.** Elhage et al. 2022. [Transformer Circuits Thread](https://transformer-circuits.pub/2022/toy_model/index.html) — Why individual neurons encode many unrelated features when a model is under-complete.
- **Towards Monosemanticity: Decomposing Language Models with Dictionary Learning.** Bricken et al. 2023. [Transformer Circuits Thread](https://transformer-circuits.pub/2023/monosemantic-features) — Sparse autoencoders recover interpretable, near-monosemantic features from superposition.
- **Sparse Autoencoders Find Highly Interpretable Features in Language Models.** Cunningham, Ewart, Riggs, Huben, Sharkey. 2023. [arXiv:2309.08600](https://arxiv.org/abs/2309.08600) — The independent, contemporaneous arXiv counterpart to Bricken et al. establishing SAEs as an unsupervised route to monosemantic features.
- **Scaling and Evaluating Sparse Autoencoders.** Gao, Dupré la Tour, Tillman, et al. (OpenAI). 2024. [arXiv:2406.04093](https://arxiv.org/abs/2406.04093) — k-sparse autoencoders with clean scaling laws in autoencoder size and sparsity, trained up to 16M latents on GPT-4 activations.
- **Sparse Autoencoders Enable Scalable and Reliable Circuit Identification.** O'Neill, Bui. 2024. [arXiv:2405.12522](https://arxiv.org/abs/2405.12522) — Uses SAE features, not raw neurons, as the units of circuit discovery.
- **Weight-sparse transformers have interpretable circuits.** Gao, Rajaram, Coxon, Govande, Baker, Mossing (OpenAI). 2025. [arXiv:2511.13653](https://arxiv.org/abs/2511.13653) — Training weights to be mostly zero yields networks whose circuits are directly human-interpretable, at a capability/interpretability trade-off — structurally relevant to protecting circuits *during* pruning rather than only analyzing them after training.
- **Decoder-Preserving Sparse Autoencoders: Which Readouts Survive Sparse Compression?** 2026. [arXiv:2607.17425](https://arxiv.org/abs/2607.17425) — Formalizes which linearly-decodable signals survive SAE-style sparse compression.
- **On the Transferability of Sparse Autoencoders for Interpreting Compressed Models.** Gupte, Chhabra, Khalili. 2025. [arXiv:2507.15977](https://arxiv.org/abs/2507.15977) — SAEs trained on the uncompressed model still interpret the compressed one, and pruning the original SAE matches training a new one — cutting the cost of auditing compressed models.

## Interpretability: Circuits

- **Zoom In: An Introduction to Circuits.** Olah, Cammarata, Schubert, Goh, Petrov, Carter. 2020. [Distill](https://distill.pub/2020/circuits/zoom-in) — Foundational statement of the circuits thesis for vision networks.
- **A Mathematical Framework for Transformer Circuits.** Elhage, Nanda, Olsson, et al. 2021. [Transformer Circuits Thread](https://transformer-circuits.pub/2021/framework/index.html) — Formal decomposition of attention-only transformers into interpretable circuit components.
- **In-context Learning and Induction Heads.** Olsson, Elhage, et al. 2022. [arXiv:2209.11895](https://arxiv.org/abs/2209.11895) — Causal evidence for induction heads as a mechanism behind in-context learning.
- **Interpretability in the Wild: A Circuit for IOI in GPT-2 small.** Wang, Variengien, Conmy, Shlegeris, Steinhardt. 2022. [arXiv:2211.00593](https://arxiv.org/abs/2211.00593) — Reverse-engineers a 26-head, 7-class circuit via causal intervention; the reference example of a fully characterized circuit.

## Interpretability: Knowledge & Function Localization

- **Knowledge Neurons in Pretrained Transformers.** Dai, Dong, Hao, Sui, Chang, Wei. 2021. [arXiv:2104.08696](https://arxiv.org/abs/2104.08696) — Localizes individual factual associations to specific MLP neurons.
- **ROME: Locating and Editing Factual Associations in GPT.** Meng, Bau, Andonian, Belinkov. 2022. [arXiv:2202.05262](https://arxiv.org/abs/2202.05262) — Causal tracing pinpoints a single fact to a mid-layer MLP, then edits it directly.
- **MEMIT: Mass-Editing Memory in a Transformer.** Meng, Sharma, Andonian, Belinkov, Bau. 2022. [arXiv:2210.07229](https://arxiv.org/abs/2210.07229) — Scales ROME-style editing to thousands of facts at once.
- **Knowledge Circuits in Pretrained Transformers.** Yao, Zhang, Xi, Wang, Xu, Deng, Chen. NeurIPS 2024. [arXiv:2405.17969](https://arxiv.org/abs/2405.17969) — Maps whole circuits (not single neurons) responsible for specific factual knowledge.
- **Function Vectors in Large Language Models.** Todd, Li, Sharma, Mueller, Wallace, Bau. 2024. [arXiv:2310.15213](https://arxiv.org/abs/2310.15213) — A single activation-space vector triggers an entire in-context task.
- **ACE: Attribution-Controlled Knowledge Editing for Multi-hop Factual Recall.** Yang, Fan, Lai, et al. 2025. [arXiv:2510.07896](https://arxiv.org/abs/2510.07896) — Neuron-level attribution locates the implicit-subject pathway used in multi-hop reasoning chains, then edits only that pathway.
- **ParamMute: Suppressing Knowledge-Critical FFNs for Faithful Retrieval-Augmented Generation.** Huang, Liu, Yan, et al. NeurIPS 2025. [arXiv:2502.15543](https://arxiv.org/abs/2502.15543) — Identifies the mid-to-deep FFN layers that are disproportionately activated when a RAG model's output contradicts the retrieved context, then suppresses those FFNs to improve faithfulness to the context.

## Interpretability: Automated Circuit Discovery & Attribution

- **ACDC: Towards Automated Circuit Discovery for Mechanistic Interpretability.** Conmy, Mavor-Parker, Lynch, Heimersheim, Garriga-Alonso. NeurIPS 2023. [arXiv:2304.14997](https://arxiv.org/abs/2304.14997) — Automates the manual activation-patching workflow behind IOI-style circuit discovery.
- **Attribution Patching Outperforms Automated Circuit Discovery.** Syed, Rager, Conmy. NeurIPS 2023 ATTRIB Workshop. [arXiv:2310.10348](https://arxiv.org/abs/2310.10348) — Linear approximation to activation patching (edge attribution patching) recovers circuits at a fraction of ACDC's cost; the method EAP-IG below refines.
- **Have Faith in Faithfulness (EAP-IG).** Hanna, Pezzelle, Belinkov. 2024. [arXiv:2403.17806](https://arxiv.org/abs/2403.17806) — Gradient-based edge attribution with integrated gradients, selected by faithfulness.
- **Sparse Feature Circuits.** Marks, Rager, Michaud, Belinkov, Bau, Mueller. 2024. [arXiv:2403.19647](https://arxiv.org/abs/2403.19647) — Discovers and edits causal graphs over SAE features rather than raw components.
- **RelP: Faithful and Efficient Circuit Discovery via Relevance Patching.** Rezaei Jafari, Eberle, Khakzar, Nanda. 2025. [arXiv:2508.21258](https://arxiv.org/abs/2508.21258) — Relevance-propagation-based patching, cheaper than gradient/activation patching at similar faithfulness.
- **MIB: A Mechanistic Interpretability Benchmark.** Mueller, Geiger, et al. 2025. [arXiv:2504.13151](https://arxiv.org/abs/2504.13151) — Standardized benchmark for comparing circuit-discovery and attribution methods.
- **PAHQ: Accelerating Automated Circuit Discovery through Mixed-Precision Inference Optimization.** Wang, Yang, Wang, Zhang, Xie, Hu, Wang. 2025. [arXiv:2510.23264](https://arxiv.org/abs/2510.23264) — Exploits an alignment between activation patching and mixed-precision quantization to speed up ACDC-style discovery.
- **Towards Understanding Fine-Tuning Mechanisms of LLMs via Circuit Analysis.** 2025. [arXiv:2502.11812](https://arxiv.org/abs/2502.11812) — Uses circuit analysis to explain *what fine-tuning changes* mechanistically, a complementary lens to compression.
- **Position-aware Automatic Circuit Discovery.** Haklay, Orgad, Bau, Mueller, Belinkov. 2025. [arXiv:2502.04577](https://arxiv.org/abs/2502.04577) — Extends edge attribution patching to distinguish token positions, improving the circuit-size/faithfulness trade-off for variable-length inputs.

## Interpretability: Verification & Faithfulness

- **Causal Scrubbing.** Chan, Garriga-Alonso, Goldowsky-Dill, Greenblatt, Nitishinskaya, Radhakrishnan, Shlegeris, Thomas. 2022. [Alignment Forum](https://www.alignmentforum.org/posts/JvZhhzycHu2Yd57RN) — General behavior-preserving-intervention procedure for testing interpretability hypotheses.
- **Hypothesis Testing the Circuit Hypothesis in LLMs.** Shi et al. NeurIPS 2024. [arXiv:2410.13032](https://arxiv.org/abs/2410.13032) — Formalizes circuit evaluation into mechanism preservation, localization, and minimality, each with a statistical test.
- **Transformer Circuit Faithfulness Metrics are not Robust.** Miller, Chughtai, Saunders. 2024. [arXiv:2407.08734](https://arxiv.org/abs/2407.08734) — Faithfulness scores are highly sensitive to seemingly minor ablation-methodology choices.
- **Adversarial Circuit Evaluation.** uit de Bos, Garriga-Alonso. 2024. [arXiv:2407.15166](https://arxiv.org/abs/2407.15166) — Adversarially searches for inputs that break a claimed circuit's faithfulness.
- **Formal Mechanistic Interpretability: Automated Circuit Discovery with Provable Guarantees.** Hadad, Katz, Bassan. 2026. [arXiv:2602.16823](https://arxiv.org/abs/2602.16823) — Brings formal-verification guarantees to circuit discovery instead of empirical faithfulness scores alone.
- **Ablation-Reversible Heads Don't Transfer: A Stress Test for Mechanistic Role Claims.** 2026. [arXiv:2606.08292](https://arxiv.org/abs/2606.08292) — Shows many claimed head-level mechanistic roles don't survive a targeted robustness stress test.
- **Certified Interventional Fidelity: Anytime-Valid, Adaptive Evaluation of Causal Claims in Mechanistic Interpretability.** Asiaee. 2026. [arXiv:2607.08349](https://arxiv.org/abs/2607.08349) — Anytime-valid confidence sequences for causal-claim fidelity, so faithfulness scores can be certified rather than read off a single point estimate.
- **Mechanistic Interpretability for AI Safety — A Review.** Bereska, Gavves. 2024. [arXiv:2404.14082](https://arxiv.org/abs/2404.14082) — Broad survey of the field, safety-application-focused.
- **A Practical Review of Mechanistic Interpretability for Transformer-Based LMs.** Rai, Zhou, Feng, Saparov, Yao. 2024. [arXiv:2407.02646](https://arxiv.org/abs/2407.02646) — Practitioner-oriented survey of tools and techniques.

## Capabilities as Manipulable Objects

- **Editing Models with Task Arithmetic.** Ilharco, Ribeiro, Wortsman, Gururangan, Schmidt, Hajishirzi, Farhadi. ICLR 2023. [arXiv:2212.04089](https://arxiv.org/abs/2212.04089) — Task vectors (fine-tuned − base weights) can be added/subtracted to add/remove capabilities.
- **From Task-Specific Models to Unified Systems: A Review of Model Merging.** Ruan, Yang, Zhou, Liu, Lu. 2025. [arXiv:2503.08998](https://arxiv.org/abs/2503.08998) — Survey of merging techniques that manipulate capabilities at the weight level.
- **Improving Alignment and Robustness with Circuit Breakers.** Zou et al. 2024. [arXiv:2406.04313](https://arxiv.org/abs/2406.04313) — Interrupts harmful internal representations mid-computation rather than filtering outputs.
- **Erasing Conceptual Knowledge from Language Models.** Gandikota et al. 2024. [arXiv:2410.02760](https://arxiv.org/abs/2410.02760) — Targeted removal of a specific concept while leaving the rest of the model intact.
- **Rethinking Machine Unlearning for Large Language Models.** Liu, Yao, Jia, et al. 2024. [arXiv:2402.08787](https://arxiv.org/abs/2402.08787) — Survey framing capability removal as a first-class LLM operation.
- **C-ΔΘ: Circuit-Restricted Weight Arithmetic for Selective Refusal.** 2026. [arXiv:2602.04521](https://arxiv.org/abs/2602.04521) — Localizes refusal to a sparse circuit via EAP-IG, then restricts a weight-arithmetic edit to only that circuit — the same localize-then-restrict pattern this list's target topic applies to compression instead of behavior editing.

## Circuit-Aware & Capability-Aware Compression (closest prior art)

*The convergence point: methods that already combine an interpretability-style signal with a compression operator. This is the most directly relevant category to selective circuit preservation during compression.*

- **TaCQ: Task-Circuit Quantization.** Xiao, Sung, Stengel-Eskin, Bansal. COLM 2025. [arXiv:2504.07389](https://arxiv.org/abs/2504.07389) — Contrasts unquantized vs. uniformly-quantized weights, uses gradients to predict per-weight task-performance impact, keeps the predicted-important weights at 16-bit. The single closest prior method; conditions on one task at a time, quantization only, no on-device or causal-faithfulness evaluation. [Code](https://github.com/The-Inscrutable-X/TACQ)
- **Attribution-Guided Pruning for Insight and Control.** Hatefi, Dreyer, Achtibat, Kahardipraja, Wiegand, Samek, Binder, Lapuschkin. 2025. [arXiv:2506.13727](https://arxiv.org/abs/2506.13727) — Layer-wise relevance propagation attributes and prunes components; removing ~0.3% of OPT-125M's neurons substantially reduces toxicity. Pruning only, small models, no causal validation. [Code](https://github.com/erfanhatefi/SparC3)
- **Capability-Guided Compression.** Gupta. 2026. [arXiv:2603.16440](https://arxiv.org/abs/2603.16440) — SAE-derived "capability density" maps set per-component compression budgets; the closest conceptual framing, but tested only on GPT-2 Medium and without joint prune+quantize or hardware evaluation.
- **Alignment-Constrained Dynamic Pruning (AAPP).** Patel, Gervacio, Raimi, Zhu, Lagasse, Grand, Panda, Chaudhary. 2025. [arXiv:2511.07482](https://arxiv.org/abs/2511.07482) — Preserves alignment-critical circuits, but per-input at inference time — by construction it never reduces stored model size.
- **Safety-Preserving PTQ via Contrastive Alignment Loss (CAQ).** Wee, Kim, Kim, Hwang, Kwak. 2025. [arXiv:2511.07842](https://arxiv.org/abs/2511.07842) — Integrates a contrastive alignment loss directly into 4-bit PTQ to protect safety behavior.
- **Preserving Fairness and Safety in Quantized LLMs Through Critical Weight Protection.** Al Hakim, Wicaksono, Koto. 2026. [arXiv:2601.12033](https://arxiv.org/abs/2601.12033) — Identifies and protects fairness- and safety-critical weights during quantization.
- **AnchorKV: Safety-Aware KV Cache Compression via Soft Penalty with a Refusal Anchor.** Ni, Lao. 2026. [arXiv:2606.17872](https://arxiv.org/abs/2606.17872) — Applies the same safety-preserving-under-compression pattern as CAQ to KV-cache eviction instead of weights, penalizing evictions that would erode refusal behavior.
- **Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications.** Wei, Huang, Huang, Xie, Qi, Xia, Mittal, Wang, Henderson. 2024. [arXiv:2402.05162](https://arxiv.org/abs/2402.05162) — Safety-critical regions can be as little as ~3% of parameters; removing them destroys safety while leaving general utility intact.
- **Finding Transformer Circuits with Edge Pruning.** Bhaskar, Wettig, Friedman, Chen. NeurIPS 2024. [arXiv:2406.16778](https://arxiv.org/abs/2406.16778) — Casts circuit discovery as edge-level optimization, scaling two orders of magnitude beyond earlier discovery methods.
- **SLiM: One-shot Quantization and Sparsity with Low-rank Approximation.** Mozaffari, Yazdanbakhsh, Mehri Dehnavi. ICML 2025. [arXiv:2410.09615](https://arxiv.org/abs/2410.09615) — Joint one-shot pruning + quantization + low-rank compensation in a single method.
- **Revisiting LLM Pruning using Neuron Semantic Attribution.** Ding, Sun, Fu, Hu. 2025. [arXiv:2503.01542](https://arxiv.org/abs/2503.01542) — Attributes neurons to semantic roles before deciding what to prune.
- **Improving Generalization in LLM Structured Pruning via Function-Aware Neuron Grouping.** Yu, An, Zhu, Zhu, Tang, Wang. 2025. [arXiv:2512.23014](https://arxiv.org/abs/2512.23014) — Groups neurons by functional role so structured pruning removes whole functions cleanly.
- **Probe Pruning: Accelerating LLMs through Dynamic Pruning via Model-Probing.** Le, Diao, Wang, Wang, Ding, Yang, Anwar. 2025. [arXiv:2502.15618](https://arxiv.org/abs/2502.15618) — Runtime probing decides what to prune per input, trading generality for adaptivity.
- **Pruning as a Domain-specific LLM Extractor.** Zhang, Liu, Zhao, Cheng, Bao, Zhang, Mitra, Chen. 2024. [arXiv:2405.06275](https://arxiv.org/abs/2405.06275) — Prunes toward a target domain's capability rather than general performance.
- **Beyond Activation Alignment: The Alignment-Diversity Tradeoff in Task-Aware LLM Quantization.** 2026. [arXiv:2607.00908](https://arxiv.org/abs/2607.00908) — Direct TaCQ-line follow-up examining what task-aware quantization trades away as tasks diversify.
- **You Had One Job: Per-Task Quantization Using LLMs' Hidden Representations.** 2025. [arXiv:2511.06516](https://arxiv.org/abs/2511.06516) — Turns mixed-precision PTQ from a model-centric step into a task-conditioned precision-allocation problem.
- **Mechanistically Interpreting Compression in Vision-Language Models.** 2026. [arXiv:2603.25035](https://arxiv.org/abs/2603.25035) — Extends the circuit-vs-compression question to the multimodal setting.
- **LLMs can Compress LLMs: Adaptive Pruning by Agents.** 2026. [arXiv:2601.09694](https://arxiv.org/abs/2601.09694) — An LLM agent adaptively selects what to prune, aiming to preserve critical knowledge pathways.

## Edge Deployment & Hardware Realization

- **A Review on Edge Large Language Models: Design, Execution, and Applications.** Zheng, Chen, Qian, Shi, Shu, Chen. 2024. [arXiv:2410.11845](https://arxiv.org/abs/2410.11845) — Survey of the full edge-LLM deployment stack.
- **Sometimes Painful but Certainly Promising: Feasibility and Trade-offs of LM Inference at the Edge.** Abstreiter, Tarkoma, Morabito. 2025. [arXiv:2503.09114](https://arxiv.org/abs/2503.09114) — Empirical accuracy/latency/memory trade-off study on real edge hardware.
- **A Systematic Evaluation of On-Device LLMs: Quantization, Performance, and Resources.** Song, Liu, Lin, et al. 2025. [arXiv:2505.15030](https://arxiv.org/abs/2505.15030) — Benchmarks quantized model families across real on-device resource budgets.
- **MobileLLM: Optimizing Sub-billion Parameter LMs for On-Device Use Cases.** Liu et al. 2024. [arXiv:2402.14905](https://arxiv.org/abs/2402.14905) — Deep-thin architecture with grouped-query attention and embedding sharing beats wider shallow nets under 350M.
- **Efficient Reasoning on the Edge.** Bondarenko, Hehn, Hesselink, et al. 2026. [arXiv:2603.16867](https://arxiv.org/abs/2603.16867) — Targets reasoning-capable inference specifically under edge compute budgets.
- **On-Device LLMs: State of the Union, 2026.** Chandra et al. 2026. [v-chandra.github.io/on-device-llms](https://v-chandra.github.io/on-device-llms/) — Living survey/roundup of the on-device LLM landscape.
- **ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware.** Cai, Zhu, Han. ICLR 2019. [arXiv:1812.00332](https://arxiv.org/abs/1812.00332) — Hardware-in-the-loop NAS without a proxy task.
- **Once-for-All: Train One Network and Specialize it for Efficient Deployment.** Cai, Gan, Wang, Zhang, Han. ICLR 2020. [arXiv:1908.09791](https://arxiv.org/abs/1908.09791) — One supernet yields specialized subnetworks per deployment target without retraining each.
- **Lightweight Software Kernels and Hardware Extensions for Sparse DNNs on Microcontrollers.** Daghero, Jahier Pagliari, Conti, Benini, Poncino, Burrello. 2025. [arXiv:2503.06183](https://arxiv.org/abs/2503.06183) — Kernel-level realization of structured sparsity on microcontroller-class hardware.
- **Efficient Edge LLM Deployment via Hessian-Aware Quantization and CPU-GPU Collaborative Inference.** 2025. [arXiv:2508.07329](https://arxiv.org/abs/2508.07329) — Splits inference across CPU/GPU with Hessian-guided precision assignment.
- **Accelerating LLM Inference with Flexible N:M Sparsity via a Fully Digital Compute-in-Memory Accelerator.** 2025. [arXiv:2504.14365](https://arxiv.org/abs/2504.14365) — Hardware realization of flexible N:M patterns directly in a compute-in-memory design.
- **Hardware Co-Design Scaling Laws via Roofline Modelling for On-Device LLMs.** Sun, Jiang, Ding, et al. 2026. [arXiv:2602.10377](https://arxiv.org/abs/2602.10377) — Combines scaling laws with roofline modeling to automate architecture co-design for on-device targets, reporting a ~19% perplexity improvement over baselines at matched hardware budget.

## Theoretical Foundations

- **Distributed Representations.** Hinton, McClelland, Rumelhart. In *Parallel Distributed Processing*, MIT Press, 1986. — The original argument for representations spread across many units.
- **Skeletonization: Trimming the Fat from a Network via Relevance Assessment.** Mozer, Smolensky. NeurIPS 1989. — Early relevance-based pruning, a precursor to Optimal Brain Damage.
- **Model Compression.** Buciluă, Caruana, Niculescu-Mizil. KDD 2006. — Coined "model compression"; ensemble knowledge compressed into one small model.
- **Predicting Parameters in Deep Learning.** Denil, Shakibi, Dinh, Ranzato, de Freitas. NeurIPS 2013. [arXiv:1306.0543](https://arxiv.org/abs/1306.0543) — Empirical evidence that deep nets are heavily over-parameterized.
- **Measuring the Intrinsic Dimension of Objective Landscapes.** Li, Farkhoor, Liu, Yosinski. ICLR 2018. [arXiv:1804.08838](https://arxiv.org/abs/1804.08838) — Defines the intrinsic dimensionality that later motivates low-rank compression.
- **Intrinsic Dimensionality Explains the Effectiveness of LM Fine-Tuning.** Aghajanyan, Gupta, Zettlemoyer. ACL 2021. [arXiv:2012.13255](https://arxiv.org/abs/2012.13255) — Fine-tuning updates live in a surprisingly low-dimensional subspace — the LoRA-family premise.
- **Deep Learning and the Information Bottleneck Principle.** Tishby, Zaslavsky. 2015. [arXiv:1503.02406](https://arxiv.org/abs/1503.02406) — Information-theoretic account of what compression inside a network discards vs. keeps.
- **The Linear Representation Hypothesis and the Geometry of LLMs.** Park, Choe, Veitch. ICML 2024. [arXiv:2311.03658](https://arxiv.org/abs/2311.03658) — Formalizes concepts as linear directions — the geometric assumption most localization/editing methods rely on.
- **Causal Abstractions of Neural Networks.** Geiger, Lu, Icard, Potts. NeurIPS 2021. — Formal framework for when a high-level causal model correctly abstracts a network's internals.
- **Investigating Gender Bias Using Causal Mediation Analysis.** Vig, Gehrmann, Belinkov, Qian, Nevo, Singer, Shieber. NeurIPS 2020. [arXiv:2004.12265](https://arxiv.org/abs/2004.12265) — Early application of causal mediation to localize behavior inside a language model.
- **Scaling Laws for Neural Language Models.** Kaplan, McCandlish, Henighan, et al. 2020. [arXiv:2001.08361](https://arxiv.org/abs/2001.08361) — Power-law relationships between compute, data, parameters, and loss.
- **Training Compute-Optimal Large Language Models (Chinchilla).** Hoffmann, Borgeaud, Mensch, et al. NeurIPS 2022. [arXiv:2203.15556](https://arxiv.org/abs/2203.15556) — Revises scaling laws toward much more data per parameter — the baseline every compression method is compared against.

---

## Scope and provenance

This list underlies a topic study on *selective circuit preservation during compression of LLMs for edge deployment*: localize the circuits behind a fixed capability profile, protect them during pruning/quantization, realize the protection in a hardware-executable form, and verify survival with circuit-faithfulness tests. Entries were gathered through iterative literature search across arXiv, conference proceedings, and interpretability-lab publications (Anthropic's Transformer Circuits Thread, OpenAI, the Alignment Forum), current as of July 2026. It is comprehensive with respect to that topic, not to LLM compression or mechanistic interpretability in general — see the much larger [Awesome-LLM-Compression](https://github.com/HuangOwen/Awesome-LLM-Compression) and interpretability-specific lists for broader coverage of either field alone.

## License

[CC0 1.0 Universal](LICENSE) — public domain. Paper titles, author lists, and links are factual metadata; use freely.
