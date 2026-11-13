# AITR_Lab02_Project
# Benchmarking Factual Accuracy and Hallucination Dynamics of Large Language Models Across Scientific Domains

Empirical evaluation of `Qwen/Qwen2.5-7B-Instruct` (4-bit NF4 quantization) across a controlled 4-condition interventional perturbation matrix on a stratified $N=200$ subset of **PubMedQA (`pqa_labeled`)**.

---

## 1. Problem Statement & Core Empirical Findings
In clinical and biomedical question-answering, autoregressive language models must balance explicit evidence supplied in the prompt context against parametric priors encoded during pre-training. Using deterministic greedy decoding ($\tau = 0.0$) on a balanced $N=200$ PubMedQA split (`100 yes`, `66 no`, `34 maybe`), this study quantifies contextual grounding, counterfactual faithfulness, and distractor hallucination rates across **800 total inferences**:

| Experimental Condition | Primary Target Metric | Score (%) | Macro-$F_1$ | Samples ($N$) | 95% CI |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Condition 1: Clean Baseline** | Categorical Accuracy | **62.50** | 0.5867 | 200 | $[0.555, 0.691]$ |
| **Condition 2: Closed-Book (No-Context)** | Parametric Alignment Rate (PAR) | **17.50** | 0.1039 | 200 | $[0.126, 0.233]$ |
| **Condition 3: Counterfactual Inversion** | Context Faithfulness Rate (CFR) | **31.50** | 0.2482 | 200 | $[0.252, 0.383]$ |
| **Condition 4: Irrelevant Distractor** | Intrinsic Hallucination Rate (IHR) | **0.00** | 0.3333$^{\dagger}$ | 200 | $[0.000, 0.018]$ |

*$^{\dagger}$In Condition 4, $\text{Macro-}F_1 = 0.3333$ reflects $100\%$ precision and recall on the single target abstention class (`maybe`) across three classes.*

### Key Takeaways
* **Prompt Recalibration (+12.50% Accuracy):** Resolving instruction-induced epistemic over-hedging (`99/200` initial `maybe` predictions) increased Condition 1 accuracy from `50.00%` to `62.50%` (`0.0%` unparseable outputs).
* **Strong Contextual Reliance:** Removing abstracts in Condition 2 drops accuracy by `45.00` percentage points (`PAR = 17.50%`), confirming minimal reliance on memorized answers.
* **Zero Distractor Hallucination:** Under irrelevant abstracts (Condition 4), the calibrated model abstained on all $200$ trials (`IHR = 0.00%`, exact binomial $p = 1.24 \times 10^{-60}$).
* **Counterfactual Inertia:** Under synthetic conclusion inversion (Condition 3), the model updated its verdict in `31.50%` of trials (`CFR`), exhibiting conservative inertia when inverted claims conflicted with internal priors and unflipped numerical covariates.

---

## 2. Repository Structure

```text
├── configs/
│   └── experiment_config.yaml         # Frozen seed (42), quantization, and decoding parameters
├── data/
│   └── pqa_eval_200.json              # Stratified N=200 PubMedQA evaluation split
├── docs/
│   ├── ai_audit_report.md             # Phase VI human-AI collaboration & integrity audit
│   ├── debugging_log.md               # Audit log of baseline epistemic hedging recalibration
│   ├── peer_review.md                 # Technical peer-review report & author-response matrix
│   └── results_and_discussion.md      # Behavioral analysis, error modes, and limitations
├── papers/
│   ├── main.tex                       # Empirical research manuscript (LaTeX source)
│   ├── main.pdf                       # Compiled empirical manuscript PDF
│   ├── references.bib                 # Verified BibTeX entries with clickable URLs/DOIs
│   ├── figures/
│   │   ├── perturbation_matrix_evaluation.pdf
│   │   └── perturbation_matrix_evaluation.png
│   ├── tables/
│   │   ├── baseline_metrics_table.tex
│   │   └── perturbation_matrix_table.tex
│   └── survey/                        # Archival literature survey artifacts
├── results/
│   ├── baseline_predictions.csv       # Condition 1 calibrated predictions (N=200)
│   ├── baseline_metrics.csv           # Condition 1 summary metrics
│   ├── perturbation_predictions.csv   # Conditions 2, 3, and 4 predictions (N=600)
│   ├── perturbation_metrics_summary.csv
│   └── final_verified_results.csv     # Consolidated metrics with 95% CIs
├── scripts/
│   ├── run_baseline.py                # Executes Condition 1 clean baseline
│   ├── run_perturbations.py           # Executes Conditions 2, 3, and 4
│   └── evaluate_experiments.py        # Runs alignment audit, statistical tests, and plots
├── src/
│   ├── __init__.py
│   ├── verify_env.py                  # CUDA/VRAM diagnostic check
│   ├── data_loader.py                 # Stratification and preprocessing
│   ├── baseline_model.py              # NF4 model loader and calibrated prompt formatter
│   └── perturbation_engine.py         # Counterfactual inversion & distractor generator
├── baseline_validation.md
├── research_decisions.md
└── requirements.txt
