import time
import torch
import logging
import os
from airllm import AutoModel
from transformers import AutoTokenizer
from huggingface_hub import scan_cache_dir

logger = logging.getLogger("LLMRouter.Loader")

class ModelLoader:
    """
    Unified loader for both Router and Specialist models.
    Handles lazy loading, memory management, and local availability checks.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.airllm_settings = config.get("airllm", {})
        self.cache = {}  # {model_id: (model, tokenizer, load_time)}
        
        self.compression = self.airllm_settings.get("compression", "4bit")
        self.cache_dir = self.airllm_settings.get("cache_dir", "./model_cache")
        logger.info(f"Initialized ModelLoader with {self.compression} compression")

    def get_local_models(self) -> dict:
        """
        Checks which models from the config are already available in the HF cache.
        Returns a dict: {model_id: bool (is_local)}
        """
        local_ids = set()
        try:
            cache_info = scan_cache_dir()
            for repo in cache_info.repos:
                local_ids.add(repo.repo_id)
        except Exception as e:
            if "Cache directory not found" not in str(e):
                logger.warning(f"Could not scan HF cache: {e}")

        all_models = [self.config["router"]["model_id"]]
        for spec in self.config["specialists"].values():
            all_models.append(spec["model_id"])
            
        return {m_id: (m_id in local_ids) for m_id in all_models}

    def get(self, model_id: str, is_router: bool = False):
        """
        Retrieves a loaded model and tokenizer.
        - Router (tiny): Uses standard transformers (fast, no weight-slicing needed).
        - Specialists (large): Uses AirLLM (memory efficient via weight-slicing).
        """
        if model_id in self.cache:
            logger.info(f"Returning cached model: {model_id}")
            return self.cache[model_id]
            
        logger.info(f"Loading model: {model_id}")
        start_time = time.time()
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            
            if is_router:
                # Standard loading for the tiny router model (0.5B fits easily)
                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype="auto",
                    device_map="auto"
                )
            else:
                # AirLLM weight-slicing for large specialists
                model = AutoModel.from_pretrained(
                    model_id,
                    compression=self.compression,
                    hf_token=None
                )
            
            load_time = time.time() - start_time
            logger.info(f"Loaded {model_id} in {load_time:.2f}s")
            
            self.cache[model_id] = (model, tokenizer, load_time)
            return self.cache[model_id]
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise

    def unload(self, model_id: str):
        """
        Unloads a model from cache and clears GPU memory.
        """
        if model_id in self.cache:
            logger.info(f"Unloading model: {model_id}")
            model, tokenizer, _ = self.cache.pop(model_id)
            
            del model
            del tokenizer
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            logger.info(f"Cleared VRAM after unloading {model_id}")
