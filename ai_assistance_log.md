# AI Assistance & Research Integrity Log (`docs/ai_assistance_log.md`)

**Project:** Benchmarking Factual Accuracy and Hallucination Dynamics of Large Language Models Across Scientific Domains  
**Author:** Veligatla Nanda Varun Gowd  
**Audit Scope:** Phases I–VI (Experimental Design, Execution, Verification, and Manuscript Drafting)

---

## 1. Governance & Attribution Policy
All empirical hypotheses, experimental parameters (`Qwen/Qwen2.5-7B-Instruct`, 4-bit NF4, seed `42`, $\tau = 0.0$), GPU execution runs, statistical audits, and final scientific claims were authored, executed, and verified directly by the human researcher. Large Language Model assistance was restricted to boilerplate code structuring, LaTeX formatting, and stylistic editing, with every quantitative figure verified against raw CSV outputs in `results/`.

---

## 2. Phase-by-Phase Collaboration Ledger

| Phase | Artifact / Task | AI Assistance Scope | Human Verification & Override | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Phase I** | Literature Synthesis & Gap Framing | Assisted in structuring taxonomy categories and formatting verified BibTeX entries (`papers/references.bib`). | Manually audited all 20+ citations, DOIs, and URLs against primary ACL, NeurIPS, AAAI, Nature, and arXiv records. | Verified |
| **Phase II** | Stratified Sampling & Model Loader (`src/data_loader.py`, `src/baseline_model.py`) | Generated boilerplate PyTorch / `bitsandbytes` NF4 initialization and stratification scaffolding. | Verified exact class counts (`100 yes`, `66 no`, `34 maybe`, seed `42`) in `data/pqa_eval_200.json` and confirmed `<5.5 GB` peak VRAM via `src/verify_env.py`. | Verified |
| **Phase III** | Baseline Prompt Recalibration (`docs/debugging_log.md`) | Suggested system prompt phrasing constraints to separate primary clinical conclusions from trailing study caveats. | Executed before/after GPU runs on T4 hardware; verified `+12.50%` accuracy gain (`50.00%` $\to$ `62.50%`) and `0.0%` unparseable outputs in `results/baseline_predictions.csv`. | Verified |
| **Phase IV** | 4-Condition Perturbation Matrix (`src/perturbation_engine.py`, `scripts/evaluate_experiments.py`) | Assisted with regex conclusion-inversion rules, cyclic derangement indexing, and `scipy` Clopper–Pearson CI script formatting. | Manually audited all `800` inference rows across `baseline_predictions.csv` and `perturbation_predictions.csv`; verified `PAR = 17.50%`, `CFR = 31.50%`, and `IHR = 0.00%` ($p = 1.24 \times 10^{-60}$). | Verified |
| **Phase V** | Two-Column ACM/IEEE Manuscript (`papers/main.tex`) | Converted figures to native TikZ/PGFPlots vector graphics and refined prose for syntactic variation and clarity. | Line-by-line verification of every table cell, equation, confidence interval, and figure coordinate against `results/final_verified_results.csv`. | Verified |
| **Phase VI** | Peer Review & Repository Sync (`docs/peer_review.md`, `README.md`, `research_decisions.md`) | Structured peer-review response matrix and repository documentation checklists. | Confirmed all boundary conditions (4-bit NF4 quantization scope, regex inversion caveats) are explicitly stated without overclaiming. | Verified |

---

## 3. Data & Metric Integrity Declaration
* **Zero Synthetic Results:** No experimental metrics, confidence intervals, or prediction logs were generated or modified by AI tools.
* **Exact Reproducibility:** Every number reported in `papers/main.tex` is deterministically reproducible from `data/pqa_eval_200.json` and `configs/experiment_config.yaml`.
