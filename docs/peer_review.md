# Technical Peer Review & Author Response Matrix (`docs/peer_review.md`)

**Manuscript:** *Benchmarking Factual Accuracy and Hallucination Dynamics of Large Language Models Across Scientific Domains: An Empirical Evaluation on Stratified PubMedQA*

---

## 1. Reviewer Concerns & Author Resolutions

| ID | Reviewer Critique | Severity | Resolution Implemented in Manuscript & Code |
| :--- | :--- | :---: | :--- |
| **R1** | **Quantization Confounding:** Findings on a 4-bit NF4 checkpoint may differ from unquantized FP16 weights due to compressed logit margins. | Major | Scoped all claims explicitly to the 4-bit NF4 regime in the Abstract, Section 1, and Section 6 (Limitations), and queued FP16/INT8 precision ablation as concrete future work. |
| **R2** | **Regex Counterfactual Boundaries:** Inverting conclusion verbs via regex leaves secondary numerical covariates ($p$-values, hazard ratios) unflipped, creating internal passage tension. | Major | Explicitly documented intra-abstract statistical tension in Section 5.3 and Table 3 as one of the two mechanisms explaining the `68.50%` conservative inertia rate (`CFR = 31.50%`). |
| **R3** | **Interpretation of Low Condition 2 & 4 Percentages:** Readers may misread `PAR = 17.50%` and `IHR = 0.00%` as poor accuracy rather than safety/grounding metrics. | Minor | Added directional arrows ($\uparrow$ / $\downarrow$) and complementary operational rates (`82.50%` Context Dependence Rate and `100.00%` Distractor Abstention Accuracy) across the Abstract, Table 2, and Figure 3. |
| **R4** | **Comparison Against Frontier Cloud APIs:** Clarify why a local 4-bit 7B model scores `62.50%` on 3-class PubMedQA compared to `>85%–90%` reported for frontier cloud APIs on binary splits. | Minor | Added Section 5.2 contrasting local `<5.5 GB` VRAM 3-class evaluation against cloud-scale binary evaluations, and provided an optional `N=166` binary-subset audit path in `scripts/evaluate_experiments.py`. |
| **R5** | **Statistical Rigor on Zero-Hallucination Finding:** Provide exact confidence bounds and significance testing for `IHR = 0.00%` (`0/200`). | Minor | Reported exact $95\%$ Clopper–Pearson confidence intervals (`[0.000, 0.018]`) and two-sided exact binomial test results ($p = 1.24 \times 10^{-60}$ vs. $p_0 = 0.50$) in Section 3.4 and Section 5.5. |
