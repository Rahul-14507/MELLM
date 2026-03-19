import time
import logging
from pathlib import Path
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

logger = logging.getLogger("LLMRouter.Loader")

# GGUF model registry — maps model_id to (repo_id, filename)
GGUF_REGISTRY = {
    "Qwen/Qwen2.5-Coder-1.5B-Instruct": (
        "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
        "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
    ),
    "Qwen/Qwen2.5-1.5B-Instruct": (
        "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "qwen2.5-1.5b-instruct-q4_k_m.gguf"
    ),
    "Qwen/Qwen2.5-Math-1.5B-Instruct": (
        "Qwen/Qwen2.5-Math-1.5B-Instruct-GGUF",
        "qwen2.5-math-1.5b-instruct-q4_k_m.gguf"
    ),
    "ContactDoctor/Bio-Medical-Llama-3-8B": (
        "bartowski/Bio-Medical-Llama-3-8B-GGUF",
        "Bio-Medical-Llama-3-8B-Q4_K_M.gguf"
    ),
    "AdaptLLM/law-LLM": (
        "AdaptLLM/law-LLM-GGUF",
        "law-llm-q4_k_m.gguf"
    ),
    "Qwen/Qwen2.5-0.5B-Instruct": (
        "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "qwen2.5-0.5b-instruct-q4_k_m.gguf"
    ),
}


class ModelLoader:
    def __init__(self, config: dict = None, compression: str = "4bit"):
        self.config = config or {}
        self.cache: dict = {}
        self.cache_dir = Path.home() / ".cache" / "mellm_gguf"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Initialized ModelLoader with llama-cpp-python (GPU inference)")

    def get_local_models(self) -> dict:
        """Returns a dict of model_id -> availability (bool)."""
        all_models = [self.config.get("router", {}).get("model_id", "")]
        for spec in self.config.get("specialists", {}).values():
            all_models.append(spec.get("model_id", ""))

        result = {}
        for m_id in all_models:
            if not m_id:
                continue
            if m_id in GGUF_REGISTRY:
                _, filename = GGUF_REGISTRY[m_id]
                result[m_id] = (self.cache_dir / filename).exists()
            else:
                result[m_id] = False
        return result

    def _get_gguf_path(self, model_id: str) -> Path:
        if model_id not in GGUF_REGISTRY:
            raise ValueError(f"No GGUF mapping found for model: {model_id}")

        repo_id, filename = GGUF_REGISTRY[model_id]
        local_path = self.cache_dir / filename

        if local_path.exists():
            logger.info(f"Found cached GGUF: {local_path}")
            return local_path

        logger.info(f"Downloading GGUF: {repo_id}/{filename}")
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(self.cache_dir)
        )
        return Path(downloaded)

    def get(self, model_id: str, is_router: bool = False):
        """
        Loads and returns a Llama model instance.
        is_router is kept for API compatibility but is unused — all models
        use the same llama-cpp-python backend.
        Returns (model, None, load_time) to maintain compatibility with orchestrator.
        """
        if model_id in self.cache:
            logger.info(f"Returning cached model: {model_id}")
            model, load_time = self.cache[model_id]
            return model, None, load_time

        logger.info(f"Loading model: {model_id}")
        start = time.time()

        gguf_path = self._get_gguf_path(model_id)

        model = Llama(
            model_path=str(gguf_path),
            n_gpu_layers=-1,   # offload all layers to GPU
            n_ctx=4096,        # context window
            n_batch=512,
            verbose=False
        )

        load_time = time.time() - start
        logger.info(f"Loaded {model_id} in {load_time:.2f}s")

        self.cache[model_id] = (model, load_time)
        return model, None, load_time

    def unload(self, model_id: str):
        if model_id in self.cache:
            logger.info(f"Unloading model: {model_id}")
            del self.cache[model_id]
            import gc
            gc.collect()
            logger.info(f"Cleared VRAM after unloading {model_id}")
