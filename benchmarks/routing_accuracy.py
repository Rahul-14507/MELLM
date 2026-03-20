"""
MELLM Routing Accuracy Benchmark
=================================
Tests the router's domain classification accuracy across 25 queries.
Run from the MELLM project root:
    python benchmarks/routing_accuracy.py

Requirements: MELLM must be set up and router model cached.
"""

import sys
import os
import time
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

# ─── Test Cases ──────────────────────────────────────────────────────────────
# 5 queries per domain, covering easy and ambiguous cases

TEST_CASES = [
    # CODE — clear programming requests
    ("Implement binary search in Java",                          "code"),
    ("Write a Python function to reverse a linked list",         "code"),
    ("Debug this segmentation fault in C++",                     "code"),
    ("What is the difference between a stack and a queue?",      "code"),
    ("Implement a hash map from scratch in Python",              "code"),

    # MATH — equations, proofs, numerical problems
    ("Solve the integral of x^2 * sin(x) dx",                   "math"),
    ("Prove that the square root of 2 is irrational",           "math"),
    ("What is the time complexity of merge sort?",               "math"),
    ("Find the eigenvalues of a 2x2 matrix",                    "math"),
    ("Differentiate f(x) = x^3 * ln(x)",                       "math"),

    # MEDICAL — health, symptoms, clinical questions
    ("What are the early warning signs of Type 2 Diabetes?",    "medical"),
    ("What causes appendicitis and how is it treated?",         "medical"),
    ("Explain how penicillin works against bacteria",           "medical"),
    ("What is the difference between Type 1 and Type 2 diabetes?", "medical"),
    ("What are the symptoms of a pulmonary embolism?",          "medical"),

    # LEGAL — law, rights, legal procedures
    ("What is the difference between civil and criminal law?",  "legal"),
    ("What is habeas corpus?",                                  "legal"),
    ("Can my employer read my work emails?",                    "legal"),
    ("What is the statute of limitations?",                     "legal"),
    ("What rights do I have if I am arrested?",                 "legal"),

    # GENERAL — philosophy, history, concepts
    ("Explain Occam's Razor with examples",                     "general"),
    ("What caused the fall of the Roman Empire?",               "general"),
    ("What is the philosophy of Stoicism?",                     "general"),
    ("Explain the concept of cognitive dissonance",             "general"),
    ("What is the difference between deductive and inductive reasoning?", "general"),
]


def run_benchmark():
    print("\n" + "=" * 60)
    print("  MELLM Routing Accuracy Benchmark")
    print("=" * 60)

    # Load config and router
    import yaml
    from pathlib import Path

    config_path = "user_config.yaml" if Path("user_config.yaml").exists() else "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    from loader.airllm_loader import ModelLoader
    from router.classifier import RouterClassifier
    from router.prompt_optimizer import PromptOptimizer

    print(f"\nLoading router model: {config['router']['model_id']}")
    loader = ModelLoader(config)
    router_model, _, load_time = loader.get(config["router"]["model_id"], is_router=True)
    print(f"Router loaded in {load_time:.2f}s\n")

    classifier = RouterClassifier()
    optimizer = PromptOptimizer()

    results = []
    correct = 0
    total = len(TEST_CASES)

    print(f"{'#':<4} {'Query':<55} {'Expected':<10} {'Got':<10} {'✓/✗'}")
    print("-" * 95)

    for i, (query, expected_domain) in enumerate(TEST_CASES, 1):
        try:
            decision = classifier.classify(router_model, None, query)
            got_domain = decision.get("domain", "unknown")
            confidence = decision.get("confidence", 0.0)
            is_correct = got_domain == expected_domain
            if is_correct:
                correct += 1
            status = "✓" if is_correct else "✗"
            query_display = query[:52] + "..." if len(query) > 52 else query
            print(f"{i:<4} {query_display:<55} {expected_domain:<10} {got_domain:<10} {status}  ({confidence:.2f})")
            results.append({
                "query": query,
                "expected": expected_domain,
                "got": got_domain,
                "confidence": confidence,
                "correct": is_correct
            })
        except Exception as e:
            print(f"{i:<4} {query[:52]:<55} {expected_domain:<10} {'ERROR':<10} ✗  ({str(e)[:30]})")
            results.append({
                "query": query,
                "expected": expected_domain,
                "got": "error",
                "confidence": 0.0,
                "correct": False
            })

    # Summary
    accuracy = correct / total * 100
    print("\n" + "=" * 60)
    print(f"  RESULTS: {correct}/{total} correct  ({accuracy:.1f}% accuracy)")
    print("=" * 60)

    # Per-domain breakdown
    print("\nPer-domain accuracy:")
    domains = ["code", "math", "medical", "legal", "general"]
    for domain in domains:
        domain_results = [r for r in results if r["expected"] == domain]
        domain_correct = sum(1 for r in domain_results if r["correct"])
        domain_total = len(domain_results)
        bar = "█" * domain_correct + "░" * (domain_total - domain_correct)
        print(f"  {domain:<10} {bar}  {domain_correct}/{domain_total}")

    # Failures
    failures = [r for r in results if not r["correct"]]
    if failures:
        print(f"\nMisclassified ({len(failures)}):")
        for f in failures:
            print(f"  \"{f['query'][:50]}\"")
            print(f"    expected={f['expected']}  got={f['got']}  confidence={f['confidence']:.2f}")

    # Save results
    output_path = "benchmarks/routing_accuracy_results.json"
    os.makedirs("benchmarks", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "results": results
        }, f, indent=2)
    print(f"\nFull results saved to {output_path}")

    loader.unload(config["router"]["model_id"])
    return accuracy


if __name__ == "__main__":
    run_benchmark()