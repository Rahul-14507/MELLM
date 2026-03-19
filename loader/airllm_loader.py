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
                logger.debug(f"[{model_id}] Router loading path detected. Using AutoModelForCausalLM.")
                from transformers import AutoModelForCausalLM
                # Use bfloat16 for stability on 3050 series if supported, or float16
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    device_map="cuda", # Force direct CUDA to avoid map overhead
                )
                logger.debug(f"[{model_id}] Router model loaded on CUDA.")
            else:
                logger.debug(f"[{model_id}] Specialist loading path detected. Using AirLLM.")
                model = AutoModel.from_pretrained(
                    model_id,
                    compression=self.compression,
                    hf_token=None
                )
                logger.debug(f"[{model_id}] AirLLM model initialized.")
            
            # --- CRITICAL PATCHES FOR TRANSFORMERS COMPATIBILITY ---
            model_class = type(model)
            if not hasattr(model_class, '_is_stateful'):
                model_class._is_stateful = False

            # Force legacy cache off globally to prevent looping
            try:
                if hasattr(model, 'config'):
                    model.config.use_cache = False  # Disable cache for inference
                    model.config.cache_implementation = None
                
                if hasattr(model, 'generation_config'):
                    model.generation_config.use_cache = False
                    model.generation_config.cache_implementation = None
                    if not is_router:
                        model.generation_config.do_sample = False
            except Exception as e:
                logger.debug(f"[{model_id}] Non-critical: Failed to apply cache patch: {e}")
            
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
