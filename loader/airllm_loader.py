import time
import torch
import logging
from airllm import AutoModel
from transformers import AutoTokenizer

logger = logging.getLogger("LLMRouter.Loader")

class ModelLoader:
    """
    Unified loader for both Router and Specialist models.
    Handles lazy loading and memory management using AirLLM.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.airllm_settings = config.get("airllm", {})
        self.cache = {}  # {model_id: (model, tokenizer, load_time)}
        
        self.compression = self.airllm_settings.get("compression", "4bit")
        self.cache_dir = self.airllm_settings.get("cache_dir", "./model_cache")
        logger.info(f"Initialized ModelLoader with {self.compression} compression")

    def get(self, model_id: str):
        """
        Retrieves a loaded model and tokenizer.
        Loads using AirLLM for memory efficiency.
        """
        if model_id in self.cache:
            logger.info(f"Returning cached model: {model_id}")
            return self.cache[model_id]
            
        logger.info(f"Loading model: {model_id}")
        start_time = time.time()
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            
            # Use AirLLM for all models to ensure they fit in VRAM during their turn
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
