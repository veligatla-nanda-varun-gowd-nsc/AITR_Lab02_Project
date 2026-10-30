# Research Decisions & Evidence Ledger (`research_decisions.md`)

## 1. Project Overview
- **Working Title:** Evaluating Context Faithfulness vs. Parametric Hallucination Dynamics in Quantized Biomedical Large Language Models [VERIFIED: from project specification]
- **Core Topic:** Benchmarking Factual Accuracy and Hallucination Dynamics of Large Language Models Across Scientific Domains [VERIFIED: from provided survey manuscript]
- **Author:** Veligatla Nanda Varun Gowd [VERIFIED]
- **Target Venue/Format:** ACM / IEEE Format (6–10 pages, implementation paper) [VERIFIED: from challenge brief]
- **Primary Benchmark:** PubMedQA (`pqa_labeled`, stratified $N = 200$) [VERIFIED: cached in `data/pqa_eval_200.json`]
- **Core Model:** `Qwen/Qwen2.5-7B-Instruct` (NF4 4-bit NormalFloat Quantization) [VERIFIED: environment audit]

## 2. Evidence Classification Standard
- **[VERIFIED]:** Directly backed by primary paper text, peer-reviewed citations, or logged terminal outputs.
- **[INFERRED]:** Research hypothesis or architectural deduction that must be proven experimentally.
- **[UNKNOWN]:** Unverified claim, untested dependency, or missing dataset split.

## 3. Milestones Tracker
- [x] Phase I: Review Scope Audit, Gap Identification & Research Brief (Gate G0 / G1) [VERIFIED]
- [x] Phase II: Environment Audit, Dataset Stratification & Baseline Verification (Gate G2) [VERIFIED]
- [ ] Phase III: Perturbation Pipeline Implementation & Interventional Experiments (Gate G3) [IN PROGRESS]
- [ ] Phase IV: Quantitative Synthesis (CFR/IHR), Manuscript Draft & Artifact Sign-Off (Gate G4)

## 4. Decision Log

| Timestamp | Phase | Decision / Action | Scientific Justification | Evidence Status |
| :--- | :--- | :--- | :--- | :--- |
| Phase I Setup | Phase I | Project workspace initialized | Alignment with Course Challenge brief and evaluation guidelines | [VERIFIED] |
| Phase I Literature | Phase I | Benchmark selection: PubMedQA (`pqa_labeled`) | Gold-standard expert-labeled biomedical QA benchmark containing categorical reasoning targets (`yes`, `no`, `maybe`) | [VERIFIED] |
| Phase II Environment | Phase II | Hardware runtime pinned to Google Colab Tesla T4 with `bitsandbytes` NF4 quantization | Accommodates 7B parameter models within 15 GB VRAM (< 5.5 GB allocated) using bfloat16 compute kernels | [VERIFIED: logged via `verify_env.py`] |
| Phase II Data | Phase II | Dataset stratification to balanced $N=200$ split (100 `yes`, 66 `no`, 34 `maybe`, Seed 42) | Eliminates majority-class guessing bias from original skewed distribution (~55% `yes`), frozen to `data/pqa_eval_200.json` | [VERIFIED: distribution logged] |
| Phase II Decoding | Phase II | Strict deterministic greedy decoding ($\tau = 0.0$, `do_sample=False`, `max_new_tokens=16`) | Guarantees bitwise reproducibility across evaluation runs and eliminates sampling variance | [VERIFIED: config locked] |
| Phase III Baseline | Phase III | Execution of direct uncalibrated zero-shot baseline (Condition 1) | Baseline yielded 50.00% accuracy and 0.4620 Macro-F1 due to severe epistemic hedging (49.5% `maybe` predictions) | [VERIFIED: logged in `baseline_metrics_run1.csv`] |
| Phase III Debugging (Prompt 11) | Phase III | System prompt recalibration for clinical conclusion extraction | Instruction alignment conflated academic disclaimers (*"further studies needed"*) with evidence absence; bounding `maybe` recovered +12.5% accuracy (62.50%) and 0.5867 Macro-F1 | [VERIFIED: logged in `baseline_metrics.csv` & `debugging_log.md`] |
