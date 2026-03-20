# Contributing to MELLM

Thanks for your interest in contributing! MELLM is a consumer-hardware MoE orchestration project and every contribution — from bug fixes to new domain specialists — is welcome.

---

## 🌟 Ways to Contribute

- 🌍 **New domain specialists** — Add science, finance, history, or any other domain
- 🧪 **Evaluation benchmarks** — Routing accuracy test suites
- 🖥️ **Web UI** — Gradio or Streamlit frontend
- 🏎️ **Performance** — Speculative decoding, batch inference, smarter caching
- 📝 **Better prompts** — Improve specialist system prompts
- 🐛 **Bug fixes** — Check the [Issues](https://github.com/Rahul-14507/MELLM/issues) tab

---

## 🚀 Development Setup

```bash
# 1. Fork and clone
git clone https://github.com/YOUR_USERNAME/MELLM
cd MELLM

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install llama-cpp-python with CUDA (required first)
pip install llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# 4. Install remaining deps
pip install -r requirements.txt

# 5. Add your HF token
echo "HF_TOKEN=your_token_here" > .env
```

---

## 🗂️ Adding a New Specialist

Follow these four steps to wire in a new domain:

### 1. Register the GGUF model

In `loader/airllm_loader.py`, add an entry to `GGUF_REGISTRY`:

```python
GGUF_REGISTRY = {
    # ...
    "your-org/your-model": (
        "gguf-repo-id/your-model-GGUF",
        "your-model.Q4_K_M.gguf"
    ),
}
```

### 2. Create a specialist class

Create `specialists/your_domain.py`:

```python
from .base_specialist import BaseSpecialist

class YourSpecialist(BaseSpecialist):
    def generate(self, prompt: str) -> str:
        response = self.model.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are an expert in YOUR DOMAIN."},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=self.max_new_tokens,
            temperature=0.7,
        )
        return self._postprocess(response["choices"][0]["message"]["content"])
```

### 3. Register in the orchestrator

In `orchestrator.py`:

```python
from specialists.your_domain import YourSpecialist

SPECIALIST_MAP = {
    # ...
    "your_domain": YourSpecialist,
}
```

### 4. Add to config and router

`config.yaml`:
```yaml
specialists:
  your_domain:
    model_id: "your-org/your-model"
    max_new_tokens: 2048
```

`router/classifier.py` — add `"your_domain"` to the domain list the router can return.

---

## 📐 Code Style

- Follow **PEP 8**; use descriptive variable names and docstrings
- Keep specialist classes focused — **one domain per file**
- Test on a **6GB VRAM GPU** before submitting (or document higher requirements)
- Do not commit `__pycache__/`, `.env`, or model cache files

---

## 🔁 Pull Request Process

1. Fork → branch (`git checkout -b feature/your-feature`)
2. Make your changes and test both **CLI** and **API** modes
3. Update `README.md` and `config.yaml` if you've added or changed models
4. Open a PR with a clear description of what changed and why

---

## 🐛 Reporting Bugs

Open an [Issue](https://github.com/Rahul-14507/MELLM/issues/new) and include:

- Your GPU model and VRAM
- Python version and OS
- The full error traceback
- Steps to reproduce

---

## 📄 License

By contributing, you agree that your work will be licensed under the project's [MIT License](LICENSE).
