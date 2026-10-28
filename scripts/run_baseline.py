"""
scripts/run_baseline.py
-----------------------
Executes Condition 1 (Clean Baseline) across the 200-sample PubMedQA subset,
logs raw predictions, and writes evaluation metrics to the results directory.
"""

import os
import sys
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import prepare_stratified_evaluation_set
from src.baseline_model import (
    get_quantized_model_and_tokenizer,
    construct_direct_prompt,
    parse_categorical_decision
)

def run_evaluation():
    os.makedirs("results", exist_ok=True)
    
    # 1. Acquire dataset
    dataset = prepare_stratified_evaluation_set()
    
    # 2. Load model
    model, tokenizer = get_quantized_model_and_tokenizer()
    
    records = []
    print(f"\n[*] Running Condition 1 (Clean Direct Baseline) over {len(dataset)} samples...")

    for item in tqdm(dataset, desc="Inference"):
        messages = construct_direct_prompt(item["abstract"], item["question"])
        prompt_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=False,        # Greedy decoding (tau = 0.0)
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id
            )

        new_tokens = output_tokens[0][inputs["input_ids"].shape[1]:]
        raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        prediction = parse_categorical_decision(raw_output)

        records.append({
            "pubid": item["pubid"],
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "raw_output": raw_output,
            "prediction": prediction,
            "is_correct": int(prediction == item["ground_truth"])
        })

    # 3. Export raw records
    df = pd.DataFrame(records)
    preds_csv_path = "results/baseline_predictions.csv"
    df.to_csv(preds_csv_path, index=False)
    print(f"\n[+] Raw predictions saved to: {preds_csv_path}")

    # 4. Compute Metrics
    valid_df = df[df["prediction"].isin(["yes", "no", "maybe"])]
    unparseable_count = len(df) - len(valid_df)

    acc = accuracy_score(df["ground_truth"], df["prediction"])
    macro_f1 = f1_score(
        df["ground_truth"], 
        df["prediction"], 
        labels=["yes", "no", "maybe"], 
        average="macro"
    )

    metrics = {
        "condition": "Condition 1 (Clean Baseline)",
        "total_samples": len(df),
        "evaluated_samples": len(valid_df),
        "unparseable_outputs": unparseable_count,
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4)
    }

    metrics_df = pd.DataFrame([metrics])
    metrics_csv_path = "results/baseline_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)

    print("\n" + "=" * 50)
    print("           BASELINE METRICS SUMMARY")
    print("=" * 50)
    print(f"Accuracy        : {acc * 100:.2f}%")
    print(f"Macro-F1        : {macro_f1 * 100:.2f}%")
    print(f"Unparseable     : {unparseable_count} / {len(df)}")
    print("=" * 50)
    print("\nClassification Report:")
    print(classification_report(df["ground_truth"], df["prediction"], labels=["yes", "no", "maybe"], digits=4))

if __name__ == "__main__":
    run_evaluation()
