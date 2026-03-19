import json
import torch
import logging
from .prompt_optimizer import PromptOptimizer

logger = logging.getLogger("LLMRouter.Classifier")

class RouterClassifier:
    """
    RouterClassifier now accepts model and tokenizer as arguments,
    making it compatible with on-demand loading.
    """
    
    VALID_DOMAINS = ["medical", "code", "math", "legal", "general"]
    
    def __init__(self):
        self.optimizer = PromptOptimizer()
        self.system_prompt = (
            "You are a specialized routing LLM. Your task is to classify user queries into one of these domains: "
            "medical, code, math, legal, general. "
            "You must also rewrite the prompt to be optimized for a domain specialist. "
            "You MUST respond ONLY with a JSON object in this exact format:\n"
            "{\n"
            "  \"domain\": \"string\",\n"
            "  \"confidence\": float,\n"
            "  \"reasoning\": \"string\",\n"
            "  \"rewritten_prompt\": \"string\"\n"
            "}"
        )

    def classify(self, model, tokenizer, user_prompt: str, device: str = "cuda") -> dict:
        """
        Classifies the user prompt using the provided model and tokenizer.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        from transformers import GenerationConfig
        gen_config = GenerationConfig(
            max_new_tokens=256
        )

        generated_ids = model.generate(
            **model_inputs,
            generation_config=gen_config,
            use_cache=False
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        try:
            clean_response = response.strip()
            if clean_response.startswith("```json"):
                clean_response = clean_response[7:]
            if clean_response.endswith("```"):
                clean_response = clean_response[:-3]
            
            decision = json.loads(clean_response.strip())
            
            if decision.get("domain") not in self.VALID_DOMAINS or decision.get("confidence", 0) < 0.6:
                return self._fallback(user_prompt)
                
            return decision
            
        except Exception as e:
            logger.error(f"Router interpretation failed: {e}")
            return self._fallback(user_prompt)

    def _fallback(self, user_prompt: str) -> dict:
        return {
            "domain": "general",
            "confidence": 0.0,
            "reasoning": "Fallback",
            "rewritten_prompt": self.optimizer.optimize("general", user_prompt)
        }
