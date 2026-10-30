import os

# 1. Create directory structure
os.makedirs("src", exist_ok=True)
os.makedirs("configs", exist_ok=True)

# 2. Create empty src/__init__.py
init_path = os.path.join("src", "__init__.py")
with open(init_path, "w", encoding="utf-8") as f:
  pass
print(f"[+] Initialized package file: {init_path}")

# 3. Create configs/experiment_config.yaml
config_content = """# ==============================================================================
# EXPERIMENTAL CONFIGURATION: Lab 02 - Context Faithfulness & Hallucination
# ==============================================================================

project:
  name: "biomedical-context-faithfulness-pubmedqa"
  benchmark: "PubMedQA"
  dataset_subset: "pqa_labeled"
  sample_size: 200
  stratification_seed: 42
  class_distribution:
    yes: 100
    no: 66
    maybe: 34

model:
  model_id: "Qwen/Qwen2.5-7B-Instruct"
  architecture: "CausalLM"
  quantization:
    load_in_4bit: true
    bnb_4bit_quant_type: "nf4"
    bnb_4bit_use_double_quant: true
    compute_dtype: "bfloat16"

inference:
  temperature: 0.0          # Strict greedy decoding (tau = 0.0)
  do_sample: false
  max_new_tokens: 16
  batch_size: 1

paths:
  data_file: "data/pqa_eval_200.json"
  results_dir: "results"
  predictions_csv: "results/baseline_predictions.csv"
  metrics_csv: "results/baseline_metrics.csv"
"""

config_path = os.path.join("configs", "experiment_config.yaml")
with open(config_path, "w", encoding="utf-8") as f:
  f.write(config_content)
print(f"[+] Created configuration file: {config_path}")
