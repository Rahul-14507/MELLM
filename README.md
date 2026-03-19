# MELLM - LLM Router

A consumer-hardware-friendly Mixture-of-Experts (MoE) orchestration system that routes queries to small, domain-specialized LLMs. Uses **llama-cpp-python** for fast GGUF inference on your GPU — the same backend as Ollama. Optimized for 6GB VRAM GPUs (like RTX 3050).

## Architecture

```
User Query -> [Load Router] -> Classify -> [Unload Router] -> [Load Specialist] -> Generate -> [Unload Specialist]
```

Models are loaded on demand per query and unloaded after to maximize VRAM headroom.

### Specialist Registry

| Domain  | Model | Format | Speed |
|---------|-------|--------|-------|
| Medical | Bio-Medical-Llama-3-8B | Q4_K_M GGUF | 8-12 t/s |
| Code    | Qwen2.5-Coder-1.5B-Instruct | Q4_K_M GGUF | 15-25 t/s |
| Math    | Qwen2.5-Math-1.5B-Instruct | Q4_K_M GGUF | 15-25 t/s |
| Legal   | AdaptLLM/law-LLM | Q4_K_M GGUF | 15-25 t/s |
| General | Qwen2.5-1.5B-Instruct | Q4_K_M GGUF | 15-25 t/s |

## Hardware Requirements

- **GPU**: NVIDIA RTX 3050 (6GB VRAM) or better with CUDA
- **RAM**: 16GB+
- **OS**: Linux with CUDA 12.1+

## Installation

> [!IMPORTANT]
> `llama-cpp-python` must be installed separately first with the CUDA wheels, then the rest of the dependencies.

**Step 1 — Install llama-cpp-python with CUDA support:**
```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

**Step 2 — Install remaining dependencies:**
```bash
pip install -r requirements.txt
```

**Step 3 — Set up `.env`:**
```bash
cp .env.example .env  # or create .env manually
# Add: HF_TOKEN=your_token_here
```

## Running

```bash
source .venv/bin/activate
python cli.py
```

On first run, GGUF models are downloaded automatically to `~/.cache/mellm_gguf/`.

## Performance

| Stage | Time |
|-------|------|
| Router load | ~1-2s |
| Router inference | ~1s |
| Specialist load (1.5B) | ~1-2s |
| Specialist load (8B medical) | ~3-5s |
| Inference (1.5B) | 15-25 tokens/s |
| Inference (8B) | 8-12 tokens/s |
| Typical end-to-end | 15-45s |

## Known Limitations

- Per-query latency from load/unload cycle.
- Medical (8B) and Legal models use more VRAM — queries may be slightly slower.
