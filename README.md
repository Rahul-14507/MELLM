# MELLM - LLM Router

A consumer-hardware-friendly Mixture-of-Experts (MoE) orchestration system that routes queries to small, domain-specialized LLMs instead of running one massive general model. Optimized for 6GB VRAM GPUs (like RTX 3050).

## Motivation
Running large 7B+ models on entry-level hardware is often slow or impossible. This system uses a tiny "Router" model to classify and rewrite queries, then swaps in domain-specialized "Specialist" models using AirLLM's weight-slicing mechanism.

## Architecture
**Both the router and specialist are loaded on demand per query and unloaded immediately after use.** This maximizes available VRAM for each model at the cost of increased per-query latency.

```
User Query -> [Load Router] -> Classify -> [Unload Router] -> [Load Specialist] -> Generate -> [Unload Specialist]
```

### Specialist Registry:
- **Medical**: Llama-3-8B (Fine-tuned for Bio-medical)
- **Code**: Qwen2.5-Coder-7B
- **Math**: Qwen2.5-Math-7B
- **Legal**: AdaptLLM-law-LLM
- **General**: Phi-3.5-mini

## Hardware Requirements
- **GPU**: NVIDIA RTX 3050 (6GB VRAM) or better
- **RAM**: 16GB+
- **CPU**: i5-13th Gen or equivalent
- **OS**: Linux/Windows with CUDA support

## Installation
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure models in `config.yaml` (default IDs are pre-configured).

## Usage
### CLI Interface
```bash
python cli.py
```

### REST API
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

## Known Limitations
- **Per-Query Latency**: Both models are loaded/unloaded for every query.
- **VRAM Constraints**: Requires 4-bit quantization (AirLLM handles this).
