"""src/perturbation_engine.py.

Biomedical perturbation generator for PubMedQA:
- Condition 2: No-Context (Closed-Book Parametric Probe)
- Condition 3: Counterfactual Inverted Context (Context Faithfulness Probe)
- Condition 4: Irrelevant Distractor Context (Groundedness / Hallucination Probe)
"""

import random
import re
from typing import Any, Dict, List


class PerturbationEngine:

  def __init__(self, seed: int = 42):
    self.seed = seed
    self.rng = random.Random(seed)

    self.inversion_patterns = [
        (
            r"\bis significantly associated with\b",
            "is not significantly associated with",
        ),
        (
            r"\bis not significantly associated with\b",
            "is significantly associated with",
        ),
        (r"\bwas significantly associated with\b", "was not associated with"),
        (r"\bwas associated with\b", "was not associated with"),
        (r"\bis associated with\b", "is not associated with"),
        (r"\bshowed significant improvement\b", "showed no improvement"),
        (r"\bdemonstrated efficacy\b", "failed to demonstrate efficacy"),
        (r"\bsignificantly reduced\b", "did not reduce"),
        (r"\bsignificantly increased\b", "did not increase"),
        (r"\bcan be recommended\b", "cannot be recommended"),
        (r"\bis effective\b", "is ineffective"),
        (r"\bis safe\b", "is unsafe"),
        (r"\bplays an important role\b", "does not play a role"),
        (r"\bcorrelates with\b", "does not correlate with"),
        (r"\bhas a beneficial effect\b", "has no beneficial effect"),
    ]

  def generate_condition_2(self, item: Dict[str, Any]) -> Dict[str, Any]:
    """Condition 2: Closed-Book (No-Context)."""
    return {
        "pubid": item["pubid"],
        "question": item["question"],
        "abstract": "",
        "ground_truth": item["ground_truth"],
        "target_comparison": item["ground_truth"],
        "condition": "Condition 2 (No-Context)",
    }

  def generate_condition_3(self, item: Dict[str, Any]) -> Dict[str, Any]:
    """Condition 3: Counterfactual Inverted Context."""
    abstract = item["abstract"]
    inverted_abstract = abstract
    for pattern, replacement in self.inversion_patterns:
      inverted_abstract = re.sub(
          pattern, replacement, inverted_abstract, flags=re.IGNORECASE
      )

    if inverted_abstract == abstract:
      inverted_abstract = (
          "CONCLUSIONS: Contrary to prior findings, results demonstrate no"
          f" significant clinical effect or validity.\n\n{abstract}"
      )

    gt = item["ground_truth"]
    inverted_target = (
        "no" if gt == "yes" else ("yes" if gt == "no" else "maybe")
    )

    return {
        "pubid": item["pubid"],
        "question": item["question"],
        "abstract": inverted_abstract,
        "ground_truth": gt,
        "target_comparison": inverted_target,
        "condition": "Condition 3 (Inverted Context)",
    }

  def generate_condition_4(
      self, dataset: List[Dict[str, Any]]
  ) -> List[Dict[str, Any]]:
    """Condition 4: Irrelevant Distractor Context."""
    n = len(dataset)
    shift = self.rng.randint(1, n - 1)
    items = []
    for i, item in enumerate(dataset):
      unrelated_item = dataset[(i + shift) % n]
      items.append({
          "pubid": item["pubid"],
          "question": item["question"],
          "abstract": unrelated_item["abstract"],
          "ground_truth": item["ground_truth"],
          "target_comparison": "maybe",
          "condition": "Condition 4 (Distractor Context)",
      })
    return items
