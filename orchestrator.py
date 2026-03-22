import time
from typing import Generator
from dotenv import load_dotenv
load_dotenv()

import yaml
import logging
from router.classifier import RouterClassifier
from router.prompt_optimizer import PromptOptimizer
from loader.airllm_loader import ModelLoader
from agents.composer import is_multi_domain, decompose_query, merge_responses

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
    - Multi-domain queries are decomposed, routed in parallel, and merged.
    - Domain continuity bias helps with short follow-ups.
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

        # Domain streak tracking — display ONLY (not used for routing)
        self.domain_streak = []         # rolling window of recent domains

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

        # Keep history very short — just topic keywords, not full responses
        context_lines = []
        for turn in self.conversation_history[-self.max_history:]:
            response_hint = turn["response"][:80].replace("\n", " ").strip()
            context_lines.append(f"- [{turn['domain'].upper()}] {turn['prompt']} → {response_hint}")

        context = "\n".join(context_lines)
        return (
            f"[CONTEXT — previous turns for reference only]\n"
            f"{context}\n"
            f"[END CONTEXT]\n\n"
            f"[NEW QUERY — classify and rewrite this]\n"
            f"{user_prompt}"
        )

    def _build_specialist_prompt(self, rewritten_prompt: str) -> str:
        """Builds a specialist prompt with the last 2 context turns (more concise than router prompt)."""
        if not self.conversation_history:
            return rewritten_prompt

        history_text = "\n".join([
            f"User: {t['prompt']}\nAssistant: {t['response'][:150]}..."
            for t in self.conversation_history[-2:]
        ])
        return (
            f"Conversation history:\n{history_text}\n\n"
            f"Current request: {rewritten_prompt}"
        )

    def _apply_domain_continuity(self, domain: str, user_prompt: str) -> str:
        """If query is a short follow-up, bias toward keeping the previous domain."""
        if not self.conversation_history:
            return domain

        prev_domain = self.conversation_history[-1]["domain"]
        word_count = len(user_prompt.strip().split())

        domain_signal_words = {
            "code": ["code", "implement", "program", "function", "class", "python",
                     "java", "javascript", "c++", "script", "algorithm"],
            "math": ["solve", "calculate", "integral", "derivative", "equation", "proof"],
            "medical": ["symptoms", "disease", "diagnosis", "treatment", "medicine"],
            "legal": ["law", "legal", "contract", "lawsuit", "rights"],
            "general": ["explain", "what is", "who is", "history", "philosophy"]
        }

        prompt_lower = user_prompt.lower()

        # Check if any explicit domain signal overrides
        for d, signals in domain_signal_words.items():
            if any(signal in prompt_lower for signal in signals):
                return domain  # explicit signal, trust the router

        # No explicit signal + short query = stay in previous domain
        if word_count <= 6 and prev_domain != domain and prev_domain != "multi-agent":
            logger.info(f"Domain continuity: short follow-up, keeping '{prev_domain}' over router's '{domain}'")
            return prev_domain

        return domain

    # ─── Multi-Agent Composition ───────────────────────────────────────────────

    def _run_multi_agent(self, user_prompt: str) -> dict:
        """
        Decomposes a multi-domain query, routes each part to the appropriate
        specialist sequentially, and merges the results into one response.
        """
        logger.info("Multi-domain query detected — activating composer...")
        sub_tasks = decompose_query(user_prompt)
        logger.info(f"Decomposed into {len(sub_tasks)} sub-tasks: {[t['domain'] for t in sub_tasks]}")

        sub_results = []
        total_inference_time = 0.0
        total_load_time = 0.0
        domains_used = []

        for task in sub_tasks:
            try:
                domain = task["domain"]
                sub_prompt = task["sub_prompt"]
                domains_used.append(domain)

                specialist_config = self.config["specialists"][domain]
                specialist_model_id = specialist_config["model_id"]

                # Hot cache check per sub-task
                if self.last_domain == domain and self.last_model is not None:
                    logger.info(f"[Composer] Cache hit for {domain} specialist")
                    model = self.last_model
                else:
                    if self.last_domain is not None and self.last_model is not None:
                        prev_model_id = self.config["specialists"][self.last_domain]["model_id"]
                        logger.info(f"[Composer] Switching from {self.last_domain} to {domain}...")
                        
                        # CRITICAL: Clear reference before unloading to allow GC
                        self.last_model = None
                        import gc
                        gc.collect()
                        
                        self.loader.unload(prev_model_id)

                    logger.info(f"[Composer] Loading {domain} specialist...")
                    start_load = time.time()
                    model, _, _ = self.loader.get(specialist_model_id)
                    total_load_time += time.time() - start_load

                    self.last_domain = domain
                    self.last_model = model

                # Run inference for this sub-task
                specialist_cls = self.SPECIALIST_MAP.get(domain, GeneralSpecialist)
                specialist = specialist_cls(
                    model=model,
                    max_new_tokens=specialist_config.get("max_new_tokens", 512)
                )

                inf_start = time.time()
                response = specialist.generate(sub_prompt)
                total_inference_time += time.time() - inf_start

                sub_results.append({
                    "domain": domain,
                    "sub_prompt": sub_prompt,
                    "response": response
                })
                logger.info(f"[Composer] {domain} specialist done.")

            except Exception as e:
                logger.error(f"[Composer] Failed to process {domain} specialist: {e}")
                sub_results.append({
                    "domain": domain,
                    "sub_prompt": sub_prompt,
                    "response": f"ERROR: Specialist {domain} failed to load or generate. {e}"
                })

        # Merge all responses into one coherent output
        merged = merge_responses(sub_results)

        # Append to conversation history as a single multi-agent turn
        self.conversation_history.append({
            "prompt": user_prompt,
            "domain": "multi-agent",
            "response": merged
        })
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]

        self.session_stats["total_queries"] += 1

        return {
            "original_prompt": user_prompt,
            "domain": "MULTI-AGENT",
            "domains_used": domains_used,
            "confidence": 1.0,
            "rewritten_prompt": f"[Composed: {' + '.join(d.upper() for d in domains_used)}]",
            "response": merged,
            "router_load_time": 0.0,
            "specialist_load_time": round(total_load_time, 2),
            "inference_time_seconds": round(total_inference_time, 2),
            "cache_hit": False,
            "context_turns": len(self.conversation_history),
            "is_multi_agent": True,
            "sub_results": sub_results,
        }

    # ─── Main Query Pipeline ──────────────────────────────────────────────────

    def stream_query(self, user_prompt: str) -> Generator[dict, None, None]:
        """
        Streaming version of query(). 
        First yields {"type": "routing", ...} with domain info,
        then yields {"type": "token", "content": "..."} for each token,
        finally yields {"type": "done", ...} with full metrics.
        """
        import time

        self.session_stats["total_queries"] += 1

        # Check for multi-domain — fall back to non-streaming for composer
        if is_multi_domain(user_prompt):
            result = self._run_multi_agent(user_prompt)
            yield {"type": "routing", "domain": result["domain"], 
                   "rewritten_prompt": result["rewritten_prompt"],
                   "is_multi_agent": True}
            yield {"type": "token", "content": result["response"]}
            yield {"type": "done", **result}
            return

        # Build contextual prompt
        contextual_prompt = self._build_contextual_prompt(user_prompt)

        # Route with persistent router
        decision = self.router_logic.classify(
            self.router_model, None, contextual_prompt
        )
        domain = decision["domain"]
        confidence = decision["confidence"]
        rewritten_prompt = decision["rewritten_prompt"]

        if confidence < 0.6:
            domain = "general"
            rewritten_prompt = self.optimizer.optimize(domain, user_prompt)

        domain = self._apply_domain_continuity(domain, user_prompt)
        self.domain_streak.append(domain)

        # Signal routing decision to caller immediately
        yield {"type": "routing", "domain": domain,
               "rewritten_prompt": rewritten_prompt,
               "confidence": confidence,
               "is_multi_agent": False}

        # Load specialist (hot cache or fresh)
        specialist_config = self.config["specialists"][domain]
        specialist_model_id = specialist_config["model_id"]
        spec_load_time = 0.0
        cache_hit = False

        if self.last_domain == domain and self.last_model is not None:
            model = self.last_model
            cache_hit = True
            self.session_stats["cache_hits"] += 1
        else:
            if self.last_domain is not None and self.last_model is not None:
                prev_model_id = self.config["specialists"][self.last_domain]["model_id"]
                # CRITICAL: Clear reference before unloading to allow GC
                self.last_model = None
                import gc
                gc.collect()
                self.loader.unload(prev_model_id)

            start_load = time.time()
            model, _, _ = self.loader.get(specialist_model_id)
            spec_load_time = time.time() - start_load
            self.last_domain = domain
            self.last_model = model

        # Signal load complete
        yield {"type": "loaded", "load_time": round(spec_load_time, 2),
               "cache_hit": cache_hit}

        # Stream tokens
        specialist_cls = self.SPECIALIST_MAP.get(domain, GeneralSpecialist)
        specialist = specialist_cls(
            model=model,
            max_new_tokens=specialist_config.get("max_new_tokens", 512)
        )

        specialist_prompt = self._build_specialist_prompt(rewritten_prompt)
        full_response = ""
        inference_start = time.time()

        for token in specialist.stream_generate(specialist_prompt):
            full_response += token
            yield {"type": "token", "content": token}

        inference_time = time.time() - inference_start

        # Update conversation history
        self.conversation_history.append({
            "prompt": user_prompt,
            "domain": domain,
            "response": full_response
        })
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]

        # Final done signal with full metrics
        yield {
            "type": "done",
            "original_prompt": user_prompt,
            "domain": domain,
            "confidence": confidence,
            "rewritten_prompt": rewritten_prompt,
            "response": full_response,
            "router_load_time": 0.0,
            "specialist_load_time": round(spec_load_time, 2),
            "inference_time_seconds": round(inference_time, 2),
            "cache_hit": cache_hit,
            "context_turns": len(self.conversation_history),
        }

    def query(self, user_prompt: str) -> dict:
        """
        Executes query pipeline:
          - Multi-domain? → Composer decomposes, routes, and merges.
          - Single domain? → Persistent Router → Domain Continuity → Streak Cache → Specialist → Generate.
        """
        # Check for multi-domain composition FIRST
        if is_multi_domain(user_prompt):
            return self._run_multi_agent(user_prompt)

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

        # Apply domain continuity bias for short follow-ups
        domain = self._apply_domain_continuity(domain, user_prompt)

        if confidence < 0.6:
            domain = "general"
            rewritten_prompt = self.optimizer.optimize(domain, user_prompt)

        # 4. Update domain streak (display-only)
        self.domain_streak.append(domain)

        # 5. Build specialist prompt with focused context (last 2 turns)
        specialist_prompt = self._build_specialist_prompt(rewritten_prompt)

        # 6. Specialist model loading/caching
        specialist_config = self.config["specialists"][domain]
        specialist_model_id = specialist_config["model_id"]

        cache_hit = False
        spec_load_time = 0.0

        try:
            if self.last_domain == domain and self.last_model is not None:
                # Exact cache hit — same domain
                logger.info(f"Cache hit — reusing {domain} specialist (no reload needed)")
                model = self.last_model
                cache_hit = True
                self.session_stats["cache_hits"] += 1

            else:
                # Full domain switch — unload previous specialist and load new one
                if self.last_domain is not None and self.last_model is not None:
                    prev_model_id = self.config["specialists"][self.last_domain]["model_id"]
                    logger.info(f"Domain switch: {self.last_domain} → {domain}, unloading previous specialist...")
                    
                    # CRITICAL: Clear reference before unloading to allow GC
                    self.last_model = None
                    import gc
                    gc.collect()
                    
                    self.loader.unload(prev_model_id)

                logger.info(f"Loading specialist model for {domain}...")
                start_load = time.time()
                model, _, _ = self.loader.get(specialist_model_id)
                spec_load_time = time.time() - start_load

                self.last_domain = domain
                self.last_model = model

            # 7. Run inference
            specialist_cls = self.SPECIALIST_MAP.get(domain, GeneralSpecialist)
            specialist = specialist_cls(
                model=model,
                max_new_tokens=specialist_config.get("max_new_tokens", 512)
            )

            inference_start = time.time()
            response_text = specialist.generate(specialist_prompt)
            inference_time = time.time() - inference_start

            # 8. Append to history
            self.conversation_history.append({
                "prompt": user_prompt,
                "domain": domain,
                "response": response_text
            })
            if len(self.conversation_history) > self.max_history:
                self.conversation_history = self.conversation_history[-self.max_history:]

            return {
                "original_prompt": user_prompt,
                "domain": domain,
                "confidence": confidence,
                "rewritten_prompt": rewritten_prompt,
                "response": response_text,
                "router_load_time": 0.0,
                "specialist_load_time": round(spec_load_time, 2),
                "inference_time_seconds": round(inference_time, 2),
                "cache_hit": cache_hit,
                "context_turns": len(self.conversation_history),
                "is_multi_agent": False,
            }

        except Exception as e:
            logger.error(f"Pipeline error for {domain}: {e}")
            return {
                "original_prompt": user_prompt,
                "domain": domain,
                "error": str(e),
                "response": f"I encountered an error while loading the specialist or generating a response: {e}"
            }

    # ─── Lifecycle ────────────────────────────────────────────────────────────

    def shutdown(self):
        """Cleanly unloads all resident models from VRAM."""
        logger.info("Shutting down — unloading all models...")
        if self.last_model is not None and self.last_domain is not None:
            self.loader.unload(self.config["specialists"][self.last_domain]["model_id"])
        self.loader.unload(self.config["router"]["model_id"])
        logger.info("VRAM cleared. Goodbye!")
