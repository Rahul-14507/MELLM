import time
import yaml
import logging
from router.classifier import RouterClassifier
from router.prompt_optimizer import PromptOptimizer
from loader.airllm_loader import ModelLoader

# Specialist imports
from specialists.medical import MedicalSpecialist
from specialists.code import CodeSpecialist
from specialists.math_specialist import MathSpecialist
from specialists.legal import LegalSpecialist
from specialists.general import GeneralSpecialist

logging.basicConfig(
    level=logging.INFO,
    format='[LLMRouter] [%(name)s] %(message)s'
)
logger = logging.getLogger("Orchestrator")

class LLMRouter:
    """
    Main pipeline entry point. Both Router and Specialist are loaded on demand.
    """
    
    SPECIALIST_MAP = {
        "medical": MedicalSpecialist,
        "code": CodeSpecialist,
        "math": MathSpecialist,
        "legal": LegalSpecialist,
        "general": GeneralSpecialist
    }
    
    def __init__(self, config_path: str = "config.yaml"):
        logger.info(f"Loading configuration from {config_path}")
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.loader = ModelLoader(self.config)
        self.router_logic = RouterClassifier()
        self.optimizer = PromptOptimizer()

    def query(self, user_prompt: str) -> dict:
        """
        Executes query: Load Router -> Classify -> Unload Router -> Load Specialist -> Generate -> Unload Specialist.
        """
        logger.info(f"Processing query: {user_prompt[:50]}...")
        
        # 1. Router Cycle
        router_model_id = self.config["router"]["model_id"]
        logger.info("Loading router model...")
        router_model, router_tokenizer, router_load_time = self.loader.get(router_model_id)
        
        decision = self.router_logic.classify(router_model, router_tokenizer, user_prompt)
        
        logger.info("Unloading router model...")
        self.loader.unload(router_model_id)
        
        # 2. Extract decision metadata
        domain = decision["domain"]
        confidence = decision["confidence"]
        rewritten_prompt = decision["rewritten_prompt"]
        
        if confidence < 0.6:
            domain = "general"
            rewritten_prompt = self.optimizer.optimize(domain, user_prompt)
            
        # 3. Specialist Cycle
        specialist_model_id = self.config["specialists"][domain]["model_id"]
        logger.info(f"Loading specialist model for {domain}...")
        spec_model, spec_tokenizer, spec_load_time = self.loader.get(specialist_model_id)
        
        specialist_cls = self.SPECIALIST_MAP.get(domain, GeneralSpecialist)
        specialist = specialist_cls(
            spec_model, 
            spec_tokenizer, 
            max_new_tokens=self.config["specialists"].get(domain, {}).get("max_new_tokens", 1024)
        )
        
        inference_start = time.time()
        response_text = specialist.generate(rewritten_prompt)
        inference_time = time.time() - inference_start
            
        logger.info(f"Unloading specialist model for {domain}...")
        self.loader.unload(specialist_model_id)
        
        return {
            "original_prompt": user_prompt,
            "domain": domain,
            "confidence": confidence,
            "rewritten_prompt": rewritten_prompt,
            "response": response_text,
            "load_time_seconds": round(router_load_time + spec_load_time, 2),
            "inference_time_seconds": round(inference_time, 2),
            "router_load_time": round(router_load_time, 2),
            "specialist_load_time": round(spec_load_time, 2)
        }
