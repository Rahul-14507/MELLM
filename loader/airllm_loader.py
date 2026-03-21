import time
import logging
import torch
from pathlib import Path
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
        "bartowski/Qwen2.5-Math-1.5B-Instruct-GGUF",
        "Qwen2.5-Math-1.5B-Instruct-Q4_K_M.gguf"
    ),
    "BioMistral/BioMistral-7B-DARE-GGUF": (
        "BioMistral/BioMistral-7B-DARE-GGUF",
        "ggml-model-Q2_K.gguf"
    ),
    "AdaptLLM/law-LLM": (
        "mradermacher/magistrate-3.2-3b-it-GGUF",
        "magistrate-3.2-3b-it.Q4_K_M.gguf"
    ),
    "Qwen/Qwen2.5-0.5B-Instruct": (
        "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "qwen2.5-0.5b-instruct-q4_k_m.gguf"
    ),
}


class ModelLoader:
    def __init__(self, config: dict = None, compression: str = "4bit"):
        self.config = config or {}
        # Merge config registry with default registry
        config_registry = self.config.get("gguf_registry", {})
        self.registry = {**GGUF_REGISTRY, **config_registry}
        
        self.cache: dict = {} # Restored for preloading/unload compatibility
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
            if m_id in self.registry:
                _, filename = self.registry[m_id]
                result[m_id] = (self.cache_dir / filename).exists()
            else:
                result[m_id] = False
        return result

    def _get_gguf_path(self, model_id: str) -> Path:
        if model_id not in self.registry:
            raise ValueError(f"No GGUF mapping found for model: {model_id}")

        repo_id, filename = self.registry[model_id]
        local_path = self.cache_dir / filename

        if local_path.exists():
            logger.info(f"Found cached GGUF: {local_path}")
            return local_path

        logger.info(f"Downloading GGUF: {repo_id}/{filename} (This may take a while...)")
        self._download_with_progress(repo_id, filename, local_path)
        return local_path

    def _download_with_progress(self, repo_id: str, filename: str, dest_path: Path) -> None:
        """Downloads a GGUF file from HuggingFace with a rich progress bar."""
        import requests
        from rich.progress import (
            Progress, DownloadColumn, BarColumn,
            TextColumn, TimeRemainingColumn, TransferSpeedColumn
        )
        from huggingface_hub import hf_hub_url
        from huggingface_hub.utils import build_hf_headers
        import os

        url = hf_hub_url(repo_id=repo_id, filename=filename)
        headers = build_hf_headers(token=os.environ.get("HF_TOKEN"))

        # Stream the download
        response = requests.get(url, headers=headers, stream=True, allow_redirects=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = dest_path.with_suffix(".tmp")

        with Progress(
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=40),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "•",
            DownloadColumn(),
            "•",
            TransferSpeedColumn(),
            "•",
            TimeRemainingColumn(),
            transient=False,
        ) as progress:
            task = progress.add_task(f"Downloading {filename}", total=total_size)

            try:
                with open(tmp_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                        if chunk:
                            f.write(chunk)
                            progress.update(task, advance=len(chunk))
            except Exception as e:
                # Clean up partial download on failure
                if tmp_path.exists():
                    tmp_path.unlink()
                raise RuntimeError(f"Download failed: {e}")

        # Rename tmp to final only after successful download
        tmp_path.rename(dest_path)
        logger.info(f"Saved to: {dest_path}")

    def get(self, model_id: str, is_router: bool = False):
        """
        Loads and returns a Llama model instance.
        Returns (model, None, load_time) to maintain compatibility with orchestrator.
        """
        if model_id in self.cache:
            logger.info(f"Returning cached model: {model_id}")
            model, load_time = self.cache[model_id]
            return model, None, load_time

        logger.info(f"Loading model: {model_id}")
        start = time.time()

        gguf_path = self._get_gguf_path(model_id)

        # In ModelLoader.get(), set n_ctx based on model file size
        # Large models (> 2GB) need a smaller context window to fit in VRAM
        import os
        file_size_gb = os.path.getsize(gguf_path) / 1e9
        n_ctx = 1024 if file_size_gb > 2.0 else 4096
        
        try:
            # For specialists (non-router), disable mmap to prevent fragmentation
            # on memory-swapping workflows. Persistent router still uses mmap.
            model = Llama(
                model_path=str(gguf_path),
                n_gpu_layers=-1,   # offload all layers to GPU
                n_ctx=n_ctx,
                n_batch=512,
                use_mmap=(is_router),  # Only mmap the persistent router
                verbose=False
            )
        except Exception as e:
            logger.error(f"Failed to initialize Llama model from {gguf_path}: {e}")
            raise RuntimeError(
                f"Model initialization failed. This often happens if the GGUF file is corrupted "
                f"or VRAM is insufficient. Try deleting the file at {gguf_path} and restarting."
            )
        
        load_time = time.time() - start
        logger.info(f"Loaded {model_id} in {load_time:.2f}s")
        
        # Keep track in cache for preloading/orchestration
        self.cache[model_id] = (model, load_time)
        return model, None, load_time

    def unload(self, model_id: str):
        if model_id in self.cache:
            logger.info(f"Unloading model: {model_id}")
            # Use pop to ensure it's removed from cache immediately
            model, _ = self.cache.pop(model_id)
            del model
            
            import gc
            gc.collect()
            
            # Aggressive VRAM release
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # wait for all CUDA ops to complete
                time.sleep(0.2)  # minimal safety buffer
                
            # Extra GC pass to be triple-sure
            gc.collect()
            
            logger.info(f"Cleared VRAM after unloading {model_id}")
