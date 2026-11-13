# Research Decisions & Evidence Ledger (`research_decisions.md`)

## 1. Project Overview
- **Core Topic:** Benchmarking Factual Accuracy and Hallucination Dynamics of Large Language Models Across Scientific Domains [VERIFIED: from approved survey manuscript]
- **Empirical Study Subtitle:** Evaluating Context Faithfulness vs. Parametric Hallucination Dynamics in Quantized Biomedical Large Language Models [VERIFIED: from project specification]
- **Author:** Veligatla Nanda Varun Gowd (Department of Information Technology, IIIT Allahabad) [VERIFIED]
- **Target Venue/Format:** ACM / IEEE Two-Column Format (implementation paper) [VERIFIED: from challenge brief]
- **Primary Benchmark:** PubMedQA (`pqa_labeled`, stratified $N = 200$, seed `42`) [VERIFIED: cached in `data/pqa_eval_200.json`]
- **Core Model:** `Qwen/Qwen2.5-7B-Instruct` (`bitsandbytes` 4-bit NormalFloat `nf4` quantization, `bfloat16` compute) [VERIFIED: environment audit]

## 2. Evidence Classification Standard
- **[VERIFIED]:** Directly backed by primary paper text, peer-reviewed citations, logged terminal outputs, or audited CSV results.
- **[INFERRED]:** Research hypothesis or architectural deduction tested experimentally.
- **[UNKNOWN]:** Unverified claim, untested precision regime (e.g., FP16/INT8), or unexecuted follow-up experiment.

## 3. Milestones Tracker
- [x] **Phase I:** Review Scope Audit, Gap Identification & Research Brief (Gate G0 / G1) [VERIFIED]
- [x] **Phase II:** Environment Audit, Dataset Stratification & Baseline Verification (Gate G2) [VERIFIED]
- [x] **Phase III:** Baseline Prompt Recalibration & Epistemic Hedging Mitigation (Gate G2 Recovery) [VERIFIED]
- [x] **Phase IV:** 4-Condition Perturbation Matrix Execution & Statistical Audit (Gate G3) [VERIFIED]
- [x] **Phase V:** LaTeX Manuscript Assembly (`papers/main.tex`, `papers/main.pdf`), Peer Review & Reproducible Docs (Gate G4) [VERIFIED]
- [x] **Phase VI:** Human–AI Collaboration & Research-Integrity Audit (`docs/ai_audit_report.md`) [VERIFIED]

## 4. Decision Log

| Timestamp | Phase | Decision / Action | Scientific Justification | Evidence Status |
| :--- | :--- | :--- | :--- | :--- |
| Phase I Setup | Phase I | Project workspace initialized with modular `src/`, `scripts/`, `configs/`, `results/`, `docs/`, and `papers/` directories | Enforces clean separation between executable pipeline code, raw CSV logs, and LaTeX manuscript artifacts | [VERIFIED] |
| Phase I Literature | Phase I | Benchmark selection: PubMedQA (`pqa_labeled`) | Gold-standard expert-labeled biomedical QA benchmark with categorical verdicts (`yes`, `no`, `maybe`) conditioned on clinical abstracts | [VERIFIED] |
| Phase II Environment | Phase II | Hardware runtime pinned to NVIDIA Tesla T4 (`15 GB` VRAM) with `bitsandbytes` 4-bit NF4 double quantization | Accommodates 7B parameter weights under `< 5.5 GB` VRAM using `bfloat16` compute kernels without API dependencies | [VERIFIED: logged via `src/verify_env.py`] |
| Phase II Data | Phase II | Dataset stratification to balanced $N=200$ split (`100 yes`, `66 no`, `34 maybe`, Seed `42`) | Eliminates majority-class guessing bias from skewed raw split (`~55% yes`), frozen to `data/pqa_eval_200.json` | [VERIFIED: distribution logged] |
| Phase II Decoding | Phase II | Deterministic greedy decoding ($\tau = 0.0$, `do_sample=False`, `max_new_tokens=16`) | Guarantees reproducibility across runs and isolates evidential shifts from stochastic sampling variance | [VERIFIED: `configs/experiment_config.yaml`] |
| Phase III Baseline | Phase III | Direct uncalibrated zero-shot baseline execution (Condition 1) | Yielded `50.00%` accuracy and `0.4620` Macro-$F_1$ due to severe epistemic hedging (`99/200` or `49.5%` `maybe` predictions) | [VERIFIED: logged in `baseline_predictions.csv`] |
| Phase III Debugging | Phase III | System prompt recalibration for clinical conclusion extraction | Uncalibrated wording conflated routine academic caveats (*"further studies warranted"*) with inconclusiveness; bounding `maybe` recovered `+12.50%` accuracy (`62.50%`) and `0.5867` Macro-$F_1$ (`0.0%` unparseable) | [VERIFIED: logged in `results/baseline_metrics.csv` & `docs/debugging_log.md`] |
| Phase IV Perturbations | Phase IV | Execution of 3 interventional regimes across $N=200$ samples ($600$ inferences): Closed-Book (Cond. 2), Counterfactual Inversion (Cond. 3), and Distractor Context (Cond. 4) | Quantifies Parametric Alignment Rate (`PAR = 17.50%`, CI `[0.126, 0.233]`), Context Faithfulness Rate (`CFR = 31.50%`, CI `[0.252, 0.383]`), and Intrinsic Hallucination Rate (`IHR = 0.00%`, CI `[0.000, 0.018]`, exact binomial $p = 1.24 \times 10^{-60}$) | [VERIFIED: logged in `results/perturbation_predictions.csv` & `results/final_verified_results.csv`] |
| Phase V Manuscript | Phase V | IEEE/ACM-style two-column LaTeX compilation (`papers/main.tex`, `papers/references.bib`) with full-width `tabularx` tables and clickable `plainurl` references | Aligns empirical findings with verified primary literature while eliminating table overflow and ensuring clickable citation traceability | [VERIFIED: compiled to `papers/main.pdf`] |
| Phase V–VI Review & Audit | Phase V–VI | Technical peer review (`docs/peer_review.md`), repository documentation (`README.md`), and human–AI collaboration ledger (`docs/ai_audit_report.md`) | Scopes claims strictly to 4-bit NF4 weights, documents regex inversion boundaries, and logs manual verification of all AI-assisted steps | [VERIFIED: Gates G0–G4 signed off] |

## 5. Open Risks & Future Work Boundaries (`[UNKNOWN]` / Planned)
- **Precision Ablation (`[UNKNOWN]`):** Full-precision `FP16` and `INT8` checkpoints have not yet been evaluated to isolate quantization-induced logit compression from base architectural inertia.
- **Semantic Counterfactual Rewriting (`[UNKNOWN]`):** Condition 3 regex conclusion inversion leaves secondary quantitative covariates ($p$-values, confidence intervals) unflipped; LLM-based full-abstract rewriting is queued as a follow-up experiment.
