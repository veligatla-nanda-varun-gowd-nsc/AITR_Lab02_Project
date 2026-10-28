"""
src/baseline_model.py
---------------------
Initializes Qwen2.5-7B-Instruct with 4-bit NF4 quantization, executes
zero-shot direct categorical inference, and extracts decisions via regex.
"""

import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

def get_quantized_model_and_tokenizer(model_id=MODEL_ID):
    """
    Loads model in 4-bit NormalFloat (NF4) with bfloat16 compute dtype.
    """
    print(f"[*] Initializing model graph: {model_id} (4-bit NF4)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    model.eval()
    return model, tokenizer

def construct_direct_prompt(abstract_text, question):
    """
    Constructs the frozen Condition 1 prompt format.
    """
    system_prompt = (
        "You are an expert biomedical information extraction system. "
        "You analyze medical abstracts with absolute scientific rigor. "
        "Your answers must be derived strictly and exclusively from the provided abstract text."
    )
    user_prompt = (
        f"Context Abstract:\n{abstract_text}\n\n"
        f"Question:\n{question}\n\n"
        'Based strictly on the abstract provided above, answer the question with exactly one word: '
        '"yes", "no", or "maybe". Do not provide additional commentary.\n'
        'Answer:'
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

def parse_categorical_decision(generated_text):
    """
    Extracts the categorical label ('yes', 'no', 'maybe') from raw output.
    """
    text = generated_text.strip().lower()
    
    # Direct start-of-line match
    match = re.search(r"\b(yes|no|maybe)\b", text)
    if match:
        return match.group(1)
    
    return "unparseable"
