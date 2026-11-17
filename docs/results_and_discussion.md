# Section IV: Empirical Results and Behavioral Analysis

## 4.1 Evaluation Overview
Across our 200-sample stratified PubMedQA partition, we evaluated the 4-bit NF4 quantized `Qwen2.5-7B-Instruct` checkpoint under four controlled evidential regimes: Clean Baseline, Closed-Book (No-Context), Inverted Counterfactual, and Irrelevant Distractor. Table 1 consolidates the resulting diagnostic metrics.

## 4.2 Parametric Prior vs. Contextual Grounding (RQ1)
Stripping the abstract in Condition 2 (No-Context) forces the model to rely exclusively on pre-trained weights. Accuracy falls from the 62.50% baseline to a Parametric Alignment Rate (PAR) of 17.50% (Macro-F1: 0.1039). That 45.00 percentage-point drop—corresponding to an 82.50% Context Dependence Rate—shows that the checkpoint reads the supplied clinical abstract to reach its answer rather than recalling memorized training examples.

## 4.3 Context Faithfulness Under Counterfactual Inversion (RQ2)
Flipping primary clinical findings and outcome qualifiers in Condition 3 yields a Context Faithfulness Rate (CFR) of 31.50% (Macro-F1: 0.2482). On the remaining 68.50% of instances, the model declined to endorse the inverted claim and defaulted to `maybe`. Quantized weights exhibit clear conservative inertia when flipped conclusion phrasing clashes with pre-trained medical priors or internal decision thresholds.

## 4.4 Intrinsic Hallucination Suppression (RQ3)
In Condition 4, replacing each study's abstract with an unrelated biomedical passage produces an Intrinsic Hallucination Rate (IHR) of 0.00% (95% CI: [0.000, 0.018], exact binomial test $p = 1.24 \times 10^{-60}$ against random binary guessing). Every single one of the 200 distractor trials resolved to `maybe` (100.00% Distractor Abstention Accuracy), showing reliable refusal and zero ungrounded assertions when presented with non-sequitur text.

# Section V: Limitations & Ranked Follow-Up Directions

### Limitations
1. **Rule-Based Perturbation Depth:** Regular-expression inversion modified primary conclusion clauses while leaving secondary quantitative covariates ($p$-values, confidence intervals) intact, occasionally introducing internal passage contradictions.
2. **Quantization Precision Effects:** Compressing weights to 4-bit NF4 narrows logit margins relative to unquantized FP16 checkpoints, which can shift boundary abstention behavior.

### Ranked Next Experiments
1. **Rank 1 (LLM-Assisted Counterfactual Rewriting):** Use a larger teacher model to rewrite entire abstracts coherently, eliminating mismatches between inverted conclusions and unflipped numerical statistics.
2. **Rank 2 (Precision Scaling: NF4 vs. INT8 vs. FP16):** Run identical perturbations across 4-bit, 8-bit, and full-precision checkpoints to separate quantization effects from base architectural inertia.
3. **Rank 3 (Logit-Lens Conflict Probing):** Track token probabilities across intermediate attention layers to pinpoint where parametric priors override conflicting context.
