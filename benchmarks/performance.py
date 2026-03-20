"""
MELLM Performance Benchmark
=============================
Measures end-to-end latency across all 5 specialist domains.
Run from the MELLM project root:
    python benchmarks/performance.py

Outputs a markdown-formatted table suitable for the README.
"""

import sys
import os
import time
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

BENCHMARK_QUERIES = {
    "code":    "Write a Python implementation of binary search with comments.",
    "math":    "Solve: find the derivative of f(x) = x^3 * sin(x) using the product rule.",
    "medical": "What are the main symptoms of Type 2 Diabetes and how is it diagnosed?",
    "legal":   "What is the difference between a civil lawsuit and a criminal prosecution?",
    "general": "Explain the philosophical concept of Occam's Razor with real-world examples.",
}

RUNS_PER_DOMAIN = 2  # first run = cold load, second run = hot cache


def run_benchmark():
    print("\n" + "=" * 65)
    print("  MELLM Performance Benchmark")
    print("=" * 65)

    import yaml
    config_path = "user_config.yaml" if Path("user_config.yaml").exists() else "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Import orchestrator
    from orchestrator import LLMRouter

    print("\nInitializing MELLM (loading persistent router)...")
    start = time.time()
    router = LLMRouter(config_path=config_path)
    router_startup = time.time() - start
    print(f"Router ready in {router_startup:.2f}s\n")

    results = {}

    for domain, query in BENCHMARK_QUERIES.items():
        print(f"\n{'─' * 65}")
        print(f"  Domain: {domain.upper()}")
        print(f"  Query:  {query[:60]}...")
        print(f"{'─' * 65}")

        domain_results = []

        for run in range(RUNS_PER_DOMAIN):
            run_label = "cold load" if run == 0 else "hot cache"
            print(f"\n  Run {run + 1} ({run_label})...")

            # Force domain switch to test cold load on run 1
            if run == 0 and router.last_domain is not None:
                router.loader.unload(
                    config["specialists"][router.last_domain]["model_id"]
                )
                router.last_domain = None
                router.last_model = None

            total_start = time.time()
            result = router.query(query)
            total_time = time.time() - total_start

            load_time = result.get("specialist_load_time", 0.0)
            inference_time = result.get("inference_time_seconds", 0.0)
            cache_hit = result.get("cache_hit", False)
            got_domain = result.get("domain", "unknown")
            response_len = len(result.get("response", "").split())

            print(f"  Routed to  : {got_domain.upper()} {'✓' if got_domain == domain else '✗ (wrong domain!)'}")
            print(f"  Load time  : {load_time:.2f}s {'(cache hit)' if cache_hit else '(fresh load)'}")
            print(f"  Inference  : {inference_time:.2f}s")
            print(f"  Total      : {total_time:.2f}s")
            print(f"  Response   : ~{response_len} words")

            domain_results.append({
                "run": run + 1,
                "label": run_label,
                "routed_to": got_domain,
                "correct_routing": got_domain == domain,
                "load_time": round(load_time, 2),
                "inference_time": round(inference_time, 2),
                "total_time": round(total_time, 2),
                "cache_hit": cache_hit,
                "response_words": response_len,
            })

        results[domain] = domain_results

    # Summary table
    print("\n\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)

    print(f"\n{'Domain':<10} {'Cold Load':<12} {'Hot Cache':<12} {'Inference':<12} {'Total (cold)'}")
    print("-" * 62)

    for domain, runs in results.items():
        cold = next((r for r in runs if r["label"] == "cold load"), None)
        hot = next((r for r in runs if r["label"] == "hot cache"), None)

        cold_load = f"{cold['load_time']:.2f}s" if cold else "—"
        hot_load = f"{hot['load_time']:.2f}s" if hot else "—"
        inference = f"{cold['inference_time']:.2f}s" if cold else "—"
        total = f"{cold['total_time']:.2f}s" if cold else "—"

        print(f"{domain:<10} {cold_load:<12} {hot_load:<12} {inference:<12} {total}")

    # README-ready markdown table
    print("\n\n--- README-ready markdown ---\n")
    print("| Domain | Cold Load | Hot Cache | Inference | Total (cold) |")
    print("|--------|-----------|-----------|-----------|--------------|")
    for domain, runs in results.items():
        cold = next((r for r in runs if r["label"] == "cold load"), None)
        hot = next((r for r in runs if r["label"] == "hot cache"), None)
        cold_load = f"~{cold['load_time']:.1f}s" if cold else "—"
        hot_load = f"**{hot['load_time']:.1f}s**" if hot else "—"
        inference = f"~{cold['inference_time']:.1f}s" if cold else "—"
        total = f"~{cold['total_time']:.1f}s" if cold else "—"
        print(f"| **{domain.capitalize()}** | {cold_load} | {hot_load} | {inference} | {total} |")

    # Save
    output_path = "benchmarks/performance_results.json"
    os.makedirs("benchmarks", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "router_startup_seconds": round(router_startup, 2),
            "gpu": _get_gpu_info(),
            "results": results
        }, f, indent=2)
    print(f"\nFull results saved to {output_path}")

    router.shutdown()


def _get_gpu_info() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
            return f"{name} ({vram}GB VRAM)"
    except Exception:
        pass
    return "Unknown GPU"


if __name__ == "__main__":
    run_benchmark()