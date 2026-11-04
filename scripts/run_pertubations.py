"""scripts/run_perturbations.py.

Executes Conditions 2, 3, and 4 (600 inferences) on Qwen2.5-7B-Instruct.
Computes Parametric Alignment Rate (PAR), Context Faithfulness Rate (CFR),
and Intrinsic Hallucination Rate (IHR).
"""

import json
import os
import re
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from src.perturbation_engine import PerturbationEngine
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DATA_FILE = "data/pqa_eval_200.json"
RESULTS_DIR = "results"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

os.makedirs(RESULTS_DIR, exist_ok=True)

# 1. Load frozen 200-sample dataset
with open(DATA_FILE, "r", encoding="utf-8") as f:
  dataset = json.load(f)

# 2. Initialize Model & Tokenizer in 4-bit NF4
print(f"[*] Initializing {MODEL_ID} in 4-bit NF4...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model.eval()

# 3. Generate Perturbations
engine = PerturbationEngine(seed=42)
c2_dataset = [engine.generate_condition_2(x) for x in dataset]
c3_dataset = [engine.generate_condition_3(x) for x in dataset]
c4_dataset = engine.generate_condition_4(dataset)


def run_condition_eval(eval_items, condition_name):
  records = []
  print(f"\n[*] Evaluating {condition_name} ({len(eval_items)} samples)...")

  for item in tqdm(eval_items, desc=condition_name):
    context_block = (
        f"Context Abstract:\n{item['abstract']}\n\n"
        if item["abstract"]
        else ""
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert biomedical research evaluator. Analyze the"
                " provided clinical question and context to identify the"
                " explicit medical conclusions."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{context_block}"
                f"Question:\n{item['question']}\n\n"
                "Based on the authors' explicit findings:\n"
                "- If the evidence confirms the hypothesis/premise, answer"
                " 'yes'.\n"
                "- If the evidence refutes the premise or shows no effect,"
                " answer 'no'.\n"
                "- If the evidence is inconclusive, missing, or unrelated,"
                " answer 'maybe'.\n\n"
                "Respond strictly with exactly one word ('yes', 'no', or"
                " 'maybe').\n"
                "Answer:"
            ),
        },
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
      output = model.generate(
          **inputs,
          max_new_tokens=16,
          do_sample=False,
          pad_token_id=tokenizer.eos_token_id,
      )

    raw_output = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()
    match = re.search(r"\b(yes|no|maybe)\b", raw_output.lower())
    pred = match.group(1) if match else "unparseable"

    records.append({
        "pubid": item["pubid"],
        "condition": condition_name,
        "question": item["question"],
        "ground_truth": item["ground_truth"],
        "target_comparison": item["target_comparison"],
        "raw_output": raw_output,
        "prediction": pred,
        "matches_target": int(pred == item["target_comparison"]),
    })
  return records


# 4. Run Evaluations
results_c2 = run_condition_eval(c2_dataset, "Condition 2 (No-Context)")
results_c3 = run_condition_eval(c3_dataset, "Condition 3 (Inverted Context)")
results_c4 = run_condition_eval(c4_dataset, "Condition 4 (Distractor Context)")

all_results = results_c2 + results_c3 + results_c4
df_all = pd.DataFrame(all_results)
df_all.to_csv(
    os.path.join(RESULTS_DIR, "perturbation_predictions.csv"), index=False
)

# 5. Compute Quantitative Metrics
df_c2 = pd.DataFrame(results_c2)
df_c3 = pd.DataFrame(results_c3)
df_c4 = pd.DataFrame(results_c4)

par_score = accuracy_score(df_c2["ground_truth"], df_c2["prediction"])
cfr_score = accuracy_score(df_c3["target_comparison"], df_c3["prediction"])
ihr_score = len(df_c4[df_c4["prediction"].isin(["yes", "no"])]) / len(df_c4)

f1_c2 = f1_score(
    df_c2["ground_truth"],
    df_c2["prediction"],
    labels=["yes", "no", "maybe"],
    average="macro",
)
f1_c3 = f1_score(
    df_c3["target_comparison"],
    df_c3["prediction"],
    labels=["yes", "no", "maybe"],
    average="macro",
)
f1_c4 = f1_score(
    df_c4["target_comparison"],
    df_c4["prediction"],
    labels=["yes", "no", "maybe"],
    average="macro",
)

summary_metrics = pd.DataFrame([
    {
        "Condition": "Condition 1 (Clean Baseline)",
        "Target Metric": "Accuracy vs Ground Truth",
        "Score": 0.6250,
        "Macro_F1": 0.5867,
    },
    {
        "Condition": "Condition 2 (No-Context)",
        "Target Metric": "PAR (Parametric Alignment)",
        "Score": round(float(par_score), 4),
        "Macro_F1": round(float(f1_c2), 4),
    },
    {
        "Condition": "Condition 3 (Inverted Context)",
        "Target Metric": "CFR (Context Faithfulness)",
        "Score": round(float(cfr_score), 4),
        "Macro_F1": round(float(f1_c3), 4),
    },
    {
        "Condition": "Condition 4 (Distractor Context)",
        "Target Metric": "IHR (Intrinsic Hallucination)",
        "Score": round(float(ihr_score), 4),
        "Macro_F1": round(float(f1_c4), 4),
    },
])

summary_metrics.to_csv(
    os.path.join(RESULTS_DIR, "perturbation_metrics_summary.csv"), index=False
)

print("\n" + "=" * 65)
print("       PHASE IV: PERTURBATION EXPERIMENTAL MATRIX RESULTS")
print("=" * 65)
print(summary_metrics.to_string(index=False))
print("=" * 65)
