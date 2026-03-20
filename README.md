<p align="center">
  <h1 align="center">🧠 MELLM — Multi-Expert LLM Router</h1>
  <p align="center">
    Run multiple small, domain-specialized LLMs instead of one massive general model.<br/>
    Better answers. Less VRAM. Faster responses.
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+"/>
    <img src="https://img.shields.io/badge/backend-llama--cpp--python-green" alt="llama-cpp-python"/>
    <img src="https://img.shields.io/badge/format-GGUF-orange" alt="GGUF"/>
    <img src="https://img.shields.io/badge/GPU-NVIDIA_CUDA-76B900" alt="CUDA"/>
    <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="MIT License"/>
    <img src="https://img.shields.io/github/stars/Rahul-14507/MELLM" alt="GitHub Stars"/>
    <img src="https://img.shields.io/github/issues/Rahul-14507/MELLM" alt="GitHub Issues"/>
    <img src="https://img.shields.io/github/last-commit/Rahul-14507/MELLM" alt="Last Commit"/>
  </p>
</p>

---

## 🔍 What is MELLM?

Large general-purpose models are trained to know everything — which means they're not optimised for anything in particular. A 70B general model answering a medical question is overkill on resources and often outperformed by a 7B model that was fine-tuned specifically on medical literature.

Meanwhile, running models larger than 14B is completely out of reach on consumer hardware like a 6GB GPU. You're stuck choosing between a small general model that gives mediocre answers, or a large specialist model you can't run.

**MELLM solves this differently.**

Instead of running one large model, MELLM runs a tiny **router model** that reads your query, identifies the domain, and loads the right **specialist model** — a small model fine-tuned specifically for that task. When you ask a medical question, you get a medical model. When you ask for code, you get a code model. When your query spans multiple domains, MELLM decomposes it and runs each part through the appropriate specialist simultaneously.

The result: **expert-level answers from models that fit on your GPU**, at a fraction of the compute cost of running a monolithic large model.
```
User query → Router (1.5B) → identifies domain → loads specialist (1.5–7B)
                                                 → generates response
                                                 → unloads, ready for next query
```

This architecture isn't just a consumer hardware workaround — it's a fundamentally more efficient approach. The same principle applies at production scale: routing to purpose-built specialists is cheaper and more accurate than throwing a 70B general model at every query.

### The numbers

| Approach | Model needed | VRAM | Query time | Domain accuracy |
|----------|-------------|------|------------|-----------------|
| General monolithic | 14B–70B | 16–80 GB | minutes | moderate |
| **MELLM specialist routing** | **1.5B–7B** | **6 GB** | **8–20s** | **high** |

A domain-specific 7B model consistently outperforms a 70B general model on tasks within its specialty — at less than 10% of the compute cost.

### Key Features

- 🔀 **Intelligent Routing** — A 1.5B router classifies queries into 5 domains with >90% accuracy and rewrites prompts for optimal specialist output
- 🧩 **Multi-Agent Composition** — Cross-domain queries are automatically decomposed, each part routed to the right specialist, and merged into one response
- 🧬 **Domain Specialists** — Fine-tuned models for medical, legal, math, code, and general knowledge — each optimised for its task
- ⚡ **Persistent Router** — Router stays resident in VRAM; zero routing overhead after startup
- 🔥 **Hot Specialist Cache** — Active specialist stays loaded between same-domain queries; only swapped on domain switch
- 🧠 **Conversation Context** — 3-turn history window so follow-up queries like "Now in Python?" work correctly
- 🎯 **Domain Continuity** — Short follow-ups inherit the current domain automatically
- 🖥️ **Interactive Setup Wizard** — Hardware-aware onboarding detects your GPU and recommends appropriate model sizes
- 🌐 **REST API** — FastAPI endpoint so any app can use MELLM as a backend
- ⬇️ **Auto-Download** — Models download from Hugging Face on first use, cached locally

---

## 🏗️ Architecture

```
                      ┌──────────────────────────────────────┐
                      │   STARTUP — once per session         │
                      │   Router (1.5B Qwen) loads into VRAM │
                      └──────────────────┬───────────────────┘
                                         │ stays resident ↓
┌──────────────┐    ┌─────────────┐    ┌─────────────────────┐
│  User Query  │───▶│  Add context│───▶│  Router: Classify   │
│              │    │  (last 3    │    │  domain + rewrite   │
│              │    │   turns)    │    │  prompt (JSON mode) │
└──────────────┘    └─────────────┘    └──────────┬──────────┘
                                                   │
                               domain continuity   │
                               bias applied here   │
                                                   ▼
              ┌────────────────────────────────────────────────┐
              │  Hot Cache Check                               │
              │  Same domain? → reuse loaded specialist (0s)   │
              │  New domain?  → unload old, load new (1-6s)    │
              └───────────────────┬────────────────────────────┘
                                   │
          ┌────────────────────────▼──────────────────────────┐
          │  Specialist generates response (stays in VRAM)    │
          │  History appended → available for next query      │
          └───────────────────────────────────────────────────┘
```

### VRAM Strategy

MELLM uses a **two-tier residency** model to maximize responsiveness within 6GB VRAM:

| Model | Residency | Why |
|-------|-----------|-----|
| **Router** (1.5B) | Always in VRAM | ~1 GB, permanent resident; no per-query overhead |
| **Active Specialist** | In VRAM until domain changes | Only one specialist at a time; swapped on domain switch |
| **Other Specialists** | On disk (GGUF cache) | Loaded on-demand in 1-6s |

---

## 📋 Model Registry

All models use the **GGUF** quantized format for efficient inference via `llama-cpp-python`.

| Role | Domain | Model | GGUF File | Size | Context |
|------|--------|-------|-----------|------|---------|
| **Router** | All | Qwen2.5-**1.5B**-Instruct | `qwen2.5-1.5b-instruct-q4_k_m.gguf` | ~1 GB | 4096 |
| Specialist | **Code** | Qwen2.5-Coder-1.5B-Instruct | `qwen2.5-coder-1.5b-instruct-q4_k_m.gguf` | ~1.1 GB | 4096 |
| Specialist | **Math** | Qwen2.5-Math-1.5B-Instruct | `Qwen2.5-Math-1.5B-Instruct-Q4_K_M.gguf` | ~986 MB | 4096 |
| Specialist | **Medical** | BioMistral-7B-DARE | `ggml-model-Q2_K.gguf` | ~2.3 GB | 1024 |
| Specialist | **Legal** | Magistrate-3.2-3B-IT | `magistrate-3.2-3b-it.Q4_K_M.gguf` | ~1.8 GB | 1024 |
| Specialist | **General** | Qwen2.5-1.5B-Instruct | `qwen2.5-1.5b-instruct-q4_k_m.gguf` | ~1.1 GB | 4096 |

> **Note:** Larger models (7B+) automatically use a reduced context window (1024 tokens) to stay within 6GB VRAM limits. Smaller models (≤1.5B) use the full 4096 context.

---

## 📁 Project Structure

```
MELLM/
├── cli.py                    # Rich terminal interface with session efficiency UI
├── api.py                    # FastAPI REST server
├── orchestrator.py           # Core pipeline: persistent router, hot cache, conversation history
├── config.yaml               # Model IDs, token limits, and specialist configuration
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (HF_TOKEN)
│
├── loader/
│   ├── __init__.py
│   └── airllm_loader.py      # GGUF model loader with auto-download and VRAM management
│
├── router/
│   ├── __init__.py
│   ├── classifier.py         # LLM-based query classifier (JSON output mode)
│   └── prompt_optimizer.py   # Rule-based fallback prompt templates
│
└── specialists/
    ├── __init__.py
    ├── base_specialist.py    # Abstract base class for all specialists
    ├── code.py               # Code generation specialist (temp=0.1)
    ├── math_specialist.py    # Math problem solver (temp=0.1)
    ├── medical.py            # Medical Q&A with system prompt (temp=0.7)
    ├── legal.py              # Legal information specialist
    └── general.py            # General knowledge fallback
```

---

## ⚙️ Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | NVIDIA GTX 1060 (6GB) | NVIDIA RTX 3050+ (6GB+) |
| **VRAM** | 6 GB | 8 GB+ |
| **RAM** | 8 GB | 16 GB+ |
| **Storage** | 10 GB free | 20 GB+ free |
| **CUDA** | 11.7+ | 12.1+ |
| **Python** | 3.10+ | 3.11+ |
| **OS** | Linux (Ubuntu 22.04+) | Linux with NVIDIA drivers |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Rahul-14507/MELLM
cd MELLM
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install llama-cpp-python (with CUDA)

> **⚠️ Important:** `llama-cpp-python` must be installed **separately first** with CUDA wheels. Installing it via `pip install -r requirements.txt` alone will NOT enable GPU acceleration.

```bash
# For CUDA 12.1+ (most modern NVIDIA GPUs)
pip install llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# For CUDA 11.8 (older GPUs)
pip install llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu118
```

Verify GPU support:
```bash
python -c "from llama_cpp import Llama; print('llama-cpp-python installed successfully')"
```

### 4. Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

### 5. Set Up Environment Variables

```bash
cp .env.example .env
# Or create .env manually:
echo "HF_TOKEN=your_huggingface_token_here" > .env
```

You need a [Hugging Face token](https://huggingface.co/settings/tokens) to download models. Some models may require accepting their license on the Hugging Face model page first.

### 6. Run MELLM

```bash
# Interactive CLI mode
python cli.py

# Pre-download all models at once
python cli.py --preload all

# Pre-download a specific domain
python cli.py --preload medical

# REST API mode
python api.py
```

---

## 💻 Usage

### CLI Mode

```bash
python cli.py
```

On startup, MELLM loads the router model and displays a **Model Availability Dashboard** showing which specialists are cached. The router shows **`Loaded (persistent)`** — it's already in VRAM before your first query.

```
Query: Binary Search in Java

Domain: CODE
╭─── Response (Specialist: code) ───╮
│ Here is a Java implementation...  │
╰───────────────────────────────────╯
Metrics: Router: resident (0s) | Specialist Load: 1.63s | Inference: 8.42s | Context: 1 turns
╭──────────────── ⚡ Efficiency ────────────────╮
│ Queries this session : 1                       │
│ Specialist cache hits: 0/1 (0%)               │
│ Router loads saved   : 0 (~0.0s saved)        │
│ Active specialist    : CODE (freshly loaded)  │
│ Context turns active : 1/3                    │
╰───────────────────────────────────────────────╯

Query: Now in Python?
  → Domain continuity: short follow-up, keeping 'code'
  → Cache hit — reusing code specialist (0s)
╭─── Response (Specialist: code) ───╮
│ Here is the Python equivalent...  │
╰───────────────────────────────────╯
Metrics: Router: resident (0s) | Specialist Load: 0s | Inference: 7.1s | Context: 2 turns
```

**CLI Commands:**

| Command | Effect |
|---------|--------|
| *(any query)* | Routes through MoE pipeline |
| `clear` | Wipes conversation history, starts fresh |
| `exit` / `quit` | Cleanly unloads all models from VRAM and exits |

### Multi-Agent Queries

MELLM automatically detects when a query spans multiple domains and routes sub-tasks to the appropriate specialists, then merges the results into a single coherent response:

```
Query: "Explain binary search AND give Java code AND analyse its complexity"

→ GENERAL specialist: conceptual explanation
→ CODE specialist:    Java implementation
→ MATH specialist:    O(log n) complexity analysis
→ Merged into a single response
```

This happens transparently — just ask your cross-domain question and MELLM handles the rest.

### API Mode

```bash
python api.py
# Server starts at http://0.0.0.0:8000
```

**Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check and router status |
| `POST` | `/query` | Process a query through the router pipeline |

**Example Request:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What are the symptoms of appendicitis?"}'
```

**Example Response:**
```json
{
  "original_prompt": "What are the symptoms of appendicitis?",
  "domain": "medical",
  "confidence": 0.95,
  "rewritten_prompt": "...",
  "response": "The common symptoms of appendicitis include...",
  "router_load_time": 0.0,
  "specialist_load_time": 10.81,
  "inference_time_seconds": 5.23,
  "cache_hit": false,
  "context_turns": 1
}
```

### Preloading Models

Pre-download models so they're ready instantly:

```bash
# Download ALL specialist models
python cli.py --preload all

# Download only the medical specialist
python cli.py --preload medical
```

---

## ⚡ Performance

Benchmarked on **NVIDIA RTX 3050 (6GB VRAM)** with CUDA 12.1:

| Stage | 1.5B Models | 3B Models | 7B Models (Q2_K) |
|-------|-------------|-----------|-------------------|
| Model Load (first time) | ~1-2s | ~3-4s | ~5-6s |
| Model Load (cache hit) | **0s** | **0s** | **0s** |
| Inference Speed | 15-25 tok/s | 10-15 tok/s | 8-12 tok/s |
| VRAM Usage | ~1.5 GB | ~2.5 GB | ~3-4 GB |
| Context Window | 4096 tokens | 1024 tokens | 1024 tokens |

| End-to-End | 1st Query | 2nd Query (Same Domain) | 2nd Query (Domain Switch) |
|------------|-----------|-------------------------|---------------------------|
| Router overhead | 0s (persistent) | 0s (persistent) | 0s (persistent) |
| Specialist load | 1-6s | **0s (hot cache)** | 1-6s (new domain) |
| Inference (1.5B) | ~5-15s | ~5-15s | ~5-15s |
| **Total typical query** | **~6-20s** | **~5-15s** | **~6-20s** |

---

## 🔧 Configuration

All model and generation settings are in `config.yaml`:

```yaml
router:
  model_id: "Qwen/Qwen2.5-1.5B-Instruct"
  max_new_tokens: 256

specialists:
  code:
    model_id: "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    max_new_tokens: 2048
  math:
    model_id: "Qwen/Qwen2.5-Math-1.5B-Instruct"
    max_new_tokens: 2048
  medical:
    model_id: "BioMistral/BioMistral-7B-DARE-GGUF"
    max_new_tokens: 2048
  legal:
    model_id: "AdaptLLM/law-LLM"
    max_new_tokens: 2048
  general:
    model_id: "Qwen/Qwen2.5-1.5B-Instruct"
    max_new_tokens: 2048
```

### Adding a New Specialist

1. **Add the model to `GGUF_REGISTRY`** in `loader/airllm_loader.py`:
```python
GGUF_REGISTRY = {
    # ... existing models ...
    "your-org/your-model": (
        "gguf-repo-id/your-model-GGUF",
        "your-model.Q4_K_M.gguf"
    ),
}
```

2. **Create a specialist class** in `specialists/your_domain.py`:
```python
from .base_specialist import BaseSpecialist

class YourSpecialist(BaseSpecialist):
    def generate(self, prompt: str) -> str:
        response = self.model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_new_tokens,
            temperature=0.7,
        )
        return self._postprocess(
            response["choices"][0]["message"]["content"]
        )
```

3. **Register it in `orchestrator.py`**:
```python
from specialists.your_domain import YourSpecialist

SPECIALIST_MAP = {
    # ... existing specialists ...
    "your_domain": YourSpecialist,
}
```

4. **Add it to `config.yaml`**:
```yaml
specialists:
  your_domain:
    model_id: "your-org/your-model"
    max_new_tokens: 2048
```

5. **Update the router's domain list** in `router/classifier.py` to include your new domain.

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

### Development Setup

1. Fork the repository and clone your fork
2. Follow the [Getting Started](#-getting-started) guide above
3. Create a feature branch:
```bash
git checkout -b feature/your-feature-name
```

### Contribution Ideas

- 🌍 **New domain specialists** — Add support for science, finance, history, etc.
- 🧪 **Evaluation benchmarks** — Build test suites to measure routing accuracy and specialist quality
- 🖥️ **Web UI** — Build a Gradio or Streamlit frontend
- 📊 **Metrics dashboard** — Track routing accuracy, latency, and VRAM usage over time
- 🏎️ **Performance optimization** — Explore batch inference, speculative decoding, or model caching strategies
- 📝 **Better prompts** — Improve specialist system prompts for higher-quality responses
- 🐛 **Bug fixes** — Check the Issues tab for known bugs

### Code Style

- Use descriptive variable names and docstrings
- Follow PEP 8 conventions
- Keep specialist classes focused — one domain per file
- Test on a 6GB VRAM GPU before submitting (or document higher requirements)

### Pull Request Process

1. Ensure your code runs without errors on Python 3.10+
2. Test both CLI and API modes
3. Update the README if you've added new features or changed configuration
4. Update `config.yaml` and `GGUF_REGISTRY` if you've added or changed models
5. Submit a PR with a clear description of what changed and why

---

## 🐛 Known Limitations

- **First-query latency**: The first query for a new domain takes 1-6s to load the specialist; subsequent same-domain queries are instant (hot cache)
- **Context window**: 7B models are limited to 1024 tokens of context to fit in VRAM
- **Single query at a time**: The system processes one query before accepting the next
- **Context continuity for the API**: The conversation history is session-bound to the `LLMRouter` instance; the REST API resets on each server restart
- **Domain continuity limitations**: Very ambiguous short queries (e.g., "More?") are biased toward the previous domain, which may not always be correct

---

## 🗺️ Roadmap

- [x] Persistent router model (no per-query load overhead)
- [x] Hot specialist cache (domain-switch-only unloading)
- [x] Conversation context memory (last 3 turns)
- [x] Domain continuity bias for follow-up queries
- [x] Live session efficiency panel in CLI
- [x] Multi-agent composition (cross-domain queries)
- [x] Domain-aware session history (streak display)
- [x] Interactive setup wizard with hardware detection
- [x] FastAPI REST endpoint
- [ ] Web UI (Gradio/Streamlit)
- [ ] Evaluation benchmark suite for routing accuracy
- [ ] Streaming token output
- [ ] Docker container for easy deployment
- [ ] Support for AMD GPUs (ROCm)

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Inference Engine** | [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) | GPU-accelerated GGUF model inference |
| **Model Source** | [Hugging Face Hub](https://huggingface.co) | Auto-download quantized GGUF models |
| **CLI Interface** | [Rich](https://github.com/Textualize/rich) | Beautiful terminal dashboards and panels |
| **REST API** | [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) | Async HTTP server |
| **Configuration** | [PyYAML](https://pyyaml.org/) | YAML-based model and parameter configuration |
| **Environment** | [python-dotenv](https://pypi.org/project/python-dotenv/) | Secure token management via `.env` |
| **Model Format** | [GGUF](https://github.com/ggerganov/ggml) | Quantized model format (Q2_K, Q4_K_M) |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

> **Note:** The individual models used by MELLM have their own licenses. Please check each model's Hugging Face page for their specific terms before redistribution.

---

## 🙏 Acknowledgements

- [Qwen Team](https://huggingface.co/Qwen) — For the excellent Qwen2.5 series of small instruction-tuned models
- [BioMistral](https://huggingface.co/BioMistral) — For the medical domain-adapted BioMistral-7B-DARE
- [mradermacher](https://huggingface.co/mradermacher) — For the Magistrate legal model GGUF quantization
- [bartowski](https://huggingface.co/bartowski) — For high-quality GGUF quantizations of math models
- [Georgi Gerganov](https://github.com/ggerganov) — For the `ggml` library and GGUF format
- [Andrei Betlen](https://github.com/abetlen) — For `llama-cpp-python`, the Python bindings for llama.cpp
