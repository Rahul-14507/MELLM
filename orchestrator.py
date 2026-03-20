import time
from dotenv import load_dotenv
load_dotenv()

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
    Main pipeline entry point.
    - Router is loaded once at startup and stays resident in VRAM.
    - Specialist model is kept hot between queries; only swapped on domain change.
    - Conversation history (last 3 turns) is prepended to each new query for context.
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
        
        # Load router once at startup — stays resident entire session
        logger.info("Loading router model (persistent)...")
        router_model_id = self.config["router"]["model_id"]
        self.router_model, _, router_load_time = self.loader.get(router_model_id, is_router=True)
        logger.info(f"Router model ready in {router_load_time:.2f}s.")
        
        # Hot specialist cache state
        self.last_domain = None
        self.last_model = None

        # Conversation history — last N turns for context awareness
        self.conversation_history = []  # list of {"prompt", "domain", "response"}
        self.max_history = 3

        # Session statistics
        self.session_stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "router_loads_saved": 0,
            "total_router_time_saved": 0.0
        }

    # ─── Conversation Context ─────────────────────────────────────────────────

    def _build_contextual_prompt(self, user_prompt: str) -> str:
        """Prepends the last N conversation turns to the user prompt for context-aware routing."""
        if not self.conversation_history:
            return user_prompt
        
        context_lines = []
        for turn in self.conversation_history[-self.max_history:]:
            # Truncate long responses to keep context concise
            response_preview = turn["response"][:300].strip()
            if len(turn["response"]) > 300:
                response_preview += "..."
            context_lines.append(
                f"User: {turn['prompt']}\n"
                f"Assistant ({turn['domain']}): {response_preview}"
            )
        
        context = "\n\n".join(context_lines)
        return (
            f"Previous conversation:\n{context}\n\n"
            f"New query: {user_prompt}"
        )

    def _build_specialist_prompt(self, rewritten_prompt: str) -> str:
        """Builds a specialist prompt with the last 2 context turns (more concise than router prompt)."""
        if not self.conversation_history:
            return rewritten_prompt
        
        context_lines = "\n".join([
            f"User: {t['prompt']}\nAssistant: {t['response'][:200]}..."
            for t in self.conversation_history[-2:]
        ])
        return (
            f"Previous conversation context:\n{context_lines}\n\n"
            f"Current request: {rewritten_prompt}"
        )

    # ─── Main Query Pipeline ──────────────────────────────────────────────────

    def query(self, user_prompt: str) -> dict:
        """
        Executes query: [Persistent Router] -> Classify (with context) -> [Hot Cache or Load Specialist] -> Generate.
        Router is always resident. Specialist is cached between same-domain queries.
        """
        logger.info(f"Processing query: {user_prompt[:50]}...")
        self.session_stats["total_queries"] += 1
        
        # 1. Build contextual prompt for routing (includes conversation history)
        contextual_prompt = self._build_contextual_prompt(user_prompt)
        
        # 2. Classify using persistent router
        decision = self.router_logic.classify(self.router_model, None, contextual_prompt)
        
        # 3. Extract decision metadata
        domain = decision["domain"]
        confidence = decision["confidence"]
        rewritten_prompt = decision["rewritten_prompt"]
        
        if confidence < 0.6:
            domain = "general"
            rewritten_prompt = self.optimizer.optimize(domain, user_prompt)
            
        # 4. Build specialist prompt with focused context (last 2 turns)
        specialist_prompt = self._build_specialist_prompt(rewritten_prompt)

        # 5. Specialist Hot Cache
        specialist_config = self.config["specialists"][domain]
        specialist_model_id = specialist_config["model_id"]
        
        cache_hit = False
        spec_load_time = 0.0
        
        if self.last_domain == domain and self.last_model is not None:
            # Cache hit — reuse the already-loaded specialist
            logger.info(f"Cache hit — reusing {domain} specialist (no reload needed)")
            model = self.last_model
            cache_hit = True
            self.session_stats["cache_hits"] += 1
        else:
            # Domain switch — unload previous specialist if one is loaded
            if self.last_domain is not None and self.last_model is not None:
                prev_model_id = self.config["specialists"][self.last_domain]["model_id"]
                logger.info(f"Domain switch: {self.last_domain} → {domain}, unloading previous specialist...")
                self.loader.unload(prev_model_id)
            
            logger.info(f"Loading specialist model for {domain}...")
            start_load = time.time()
            model, _, _ = self.loader.get(specialist_model_id)
            spec_load_time = time.time() - start_load
            
            self.last_domain = domain
            self.last_model = model
        
        # 6. Run inference with the specialist (context-enriched prompt)
        specialist_cls = self.SPECIALIST_MAP.get(domain, GeneralSpecialist)
        specialist = specialist_cls(
            model=model,
            max_new_tokens=specialist_config.get("max_new_tokens", 512)
        )
        
        inference_start = time.time()
        response_text = specialist.generate(specialist_prompt)
        inference_time = time.time() - inference_start
        
        # 7. Append this turn to conversation history
        self.conversation_history.append({
            "prompt": user_prompt,       # original, not contextual
            "domain": domain,
            "response": response_text
        })
        # Trim to max_history turns
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
        
        # DO NOT unload specialist — keep hot for next query
        
        return {
            "original_prompt": user_prompt,
            "domain": domain,
            "confidence": confidence,
            "rewritten_prompt": rewritten_prompt,
            "response": response_text,
            "router_load_time": 0.0,  # Router is always resident
            "specialist_load_time": round(spec_load_time, 2),
            "inference_time_seconds": round(inference_time, 2),
            "cache_hit": cache_hit,
            "context_turns": len(self.conversation_history),
        }

    # ─── Lifecycle ────────────────────────────────────────────────────────────

    def shutdown(self):
        """Cleanly unloads all resident models from VRAM."""
        logger.info("Shutting down — unloading all models...")
        if self.last_model is not None and self.last_domain is not None:
            self.loader.unload(self.config["specialists"][self.last_domain]["model_id"])
        self.loader.unload(self.config["router"]["model_id"])
        logger.info("VRAM cleared. Goodbye!")
