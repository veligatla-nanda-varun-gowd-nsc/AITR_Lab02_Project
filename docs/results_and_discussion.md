# Section IV: Empirical Results and Behavioral Analysis

## 4.1 Evaluation Overview
We evaluated the 4-bit NF4 quantized `Qwen2.5-7B-Instruct` model across a stratified 200-sample PubMedQA benchmark under four distinct evidential regimes: Clean Baseline, Closed-Book (No-Context), Inverted Counterfactual, and Irrelevant Distractor. Table 1 summarizes the primary diagnostic scores.

## 4.2 Parametric Prior vs. Contextual Grounding (RQ1)
In Condition 2 (No-Context), the abstract was completely omitted, forcing the model to rely solely on internal parametric weights. Performance collapsed from 62.50% baseline accuracy to a Parametric Alignment Rate (PAR) of 17.50% (Macro-F1: 0.1039). This steep degradation ($\Delta = -45.00\%$) confirms that the model does not exploit memorization priors to solve PubMedQA, but rather grounds its categorical decisions in the provided context abstract.

## 4.3 Context Faithfulness Under Counterfactual Inversion (RQ2)
In Condition 3, key clinical findings and outcome qualifiers were systematically inverted. The model achieved a Context Faithfulness Rate (CFR) of 31.50% (Macro-F1: 0.2482). In the remaining 68.50% of cases, the model resisted the counterfactual context—frequently defaulting to `maybe` rather than endorsing the inverted claim. This demonstrates that quantized representations maintain conservative inertia against syntactic inversions when internal safety thresholds or token associations conflict with the prompt.

## 4.4 Intrinsic Hallucination Suppression (RQ3)
In Condition 4, genuine abstracts were swapped with topically unrelated biomedical abstracts. The model achieved an Intrinsic Hallucination Rate (IHR) of 0.00% (95% CI: [0.000, 0.018], exact binomial test $p = 1.24 \times 10^{-60}$ against random guessing). Across all 200 distractor trials, the model output `maybe` in 100% of samples, demonstrating effective abstention behavior and zero ungrounded assertions under non-sequitur evidence.

# Section V: Limitations & Ranked Follow-Up Directions

### Limitations
1. **Rule-Based Perturbation Depth:** Inversion targeted primary conclusion clauses but did not perturb secondary quantitative covariates (e.g., $p$-values, confidence intervals), occasionally creating subtle within-document inconsistencies.
2. **Quantization Precision Effects:** 4-bit NF4 quantization alters margin boundaries relative to unquantized FP16 baselines.

### Ranked Next Experiments
1. **Rank 1 (LLM-Assisted Counterfactual Rewriting):** Generate holistic, end-to-end synthetic counterfactual abstracts using a larger teacher model to eliminate lexical artifacts.
2. **Rank 2 (Precision Scaling: NF4 vs. INT8 vs. FP16):** Assess whether the low CFR (31.50%) is driven by quantization error or core model architecture.
3. **Rank 3 (Logit-Lens Conflict Probing):** Measure hidden-state token probabilities at attention heads during evidence conflict.
