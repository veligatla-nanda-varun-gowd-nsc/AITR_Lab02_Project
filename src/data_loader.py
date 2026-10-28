"""
src/data_loader.py
------------------
Loads PubMedQA (pqa_labeled), formats abstract contexts,
extracts a stratified 200-sample evaluation subset, and caches to disk.
"""

import json
import os
from collections import Counter
from datasets import load_dataset

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "pqa_eval_200.json")

def format_abstract(context_dict):
    """
    Concatenates labeled abstract sections into a coherent biomedical text block.
    """
    labels = context_dict.get("labels", [])
    contexts = context_dict.get("contexts", [])
    
    sections = []
    for label, text in zip(labels, contexts):
        clean_text = text.strip()
        if label and label.strip():
            sections.append(f"{label.strip()}: {clean_text}")
        else:
            sections.append(clean_text)
    return "\n\n".join(sections)

def prepare_stratified_evaluation_set(output_path=DATA_PATH, seed=42):
    """
    Constructs a balanced 200-sample evaluation split:
    - 100 'yes'
    - 66 'no'
    - 34 'maybe'
    """
    if os.path.exists(output_path):
        print(f"[+] Loaded existing cached dataset: {output_path}")
        with open(output_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print("[*] Downloading PubMedQA ('pqa_labeled') from Hugging Face...")
    ds = load_dataset("pubmed_qa", "pqa_labeled", split="train")

    class_buckets = {"yes": [], "no": [], "maybe": []}

    for item in ds:
        label = item["final_decision"].strip().lower()
        if label in class_buckets:
            abstract_text = format_abstract(item["context"])
            class_buckets[label].append({
                "pubid": int(item["pubid"]),
                "question": item["question"].strip(),
                "abstract": abstract_text,
                "ground_truth": label,
                "long_answer": item.get("long_answer", "").strip()
            })

    # Shuffle deterministically
    import random
    rng = random.Random(seed)
    rng.shuffle(class_buckets["yes"])
    rng.shuffle(class_buckets["no"])
    rng.shuffle(class_buckets["maybe"])

    stratified_subset = (
        class_buckets["yes"][:100] +
        class_buckets["no"][:66] +
        class_buckets["maybe"][:34]
    )
    rng.shuffle(stratified_subset)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stratified_subset, f, indent=2)

    counts = Counter([x["ground_truth"] for x in stratified_subset])
    print(f"[+] Stratified dataset saved ({len(stratified_subset)} samples): {counts}")
    return stratified_subset

if __name__ == "__main__":
    prepare_stratified_evaluation_set()
