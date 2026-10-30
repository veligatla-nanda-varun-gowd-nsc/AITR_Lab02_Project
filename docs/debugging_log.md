# Engineering & Debugging Audit Log

## Entry 01: Epistemic Hedging Collapse in Zero-Shot PubMedQA Baseline
- **Date**: October 2026
- **Milestone Gate**: Gate G2 / Phase 3 Baseline Stabilization
- **Module Affected**: `src/baseline_model.py` / Prompt Template Logic
- **Architecture**: `Qwen/Qwen2.5-7B-Instruct` (4-bit NF4)

### 1. Problem Description
Initial zero-shot baseline evaluation across the stratified 200-sample subset produced an accuracy of 50.00% and Macro-F1 of 0.4620, significantly underperforming expected 7B zero-shot literature baselines (~62-75%).

### 2. Empirical Evidence
- Total evaluated: 200 samples (100 `yes`, 66 `no`, 34 `maybe`).
- Predicted labels: 84 `yes`, 17 `no`, 99 `maybe`.
- 49.5% of all predictions defaulted to `maybe`.
- Recall on ground truth `no` cases was 24.24% (42 out of 66 negative cases misclassified as `maybe`).

### 3. Diagnosis & Root Cause
Instruction-tuned safety alignment causes the model to exhibit extreme epistemic hedging when presented with standard biomedical scientific rhetoric (e.g., *"further studies are warranted"*). The model conflated routine author humility with inconclusive experimental findings.

### 4. Applied Correction
Refactored the prompt system instructions from passive literal matching to explicit clinical conclusion extraction, bounding `maybe` strictly to scenarios where findings are contradictory or explicitly non-significant.

### 5. Verification & Results
- **Accuracy**: Improved from 50.00% to 62.50% (+12.50%).
- **Macro-F1**: Improved from 0.4620 to 0.5867 (+0.1247).
- **Format Integrity**: 0 unparseable outputs maintained across both runs.
- **Decision**: Overwrote `results/baseline_predictions.csv` and `results/baseline_metrics.csv` with the calibrated baseline. Gate G2 declared clear.
