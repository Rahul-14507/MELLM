import json
import logging
from .prompt_optimizer import PromptOptimizer

logger = logging.getLogger("LLMRouter.Classifier")


class RouterClassifier:
    """
    Routes queries to the correct specialist domain using a Llama model.
    llama-cpp-python handles chat templating internally — no tokenizer needed.
    """

    VALID_DOMAINS = ["medical", "code", "math", "legal", "general"]

    def __init__(self):
        self.optimizer = PromptOptimizer()
        self.system_prompt = (
            "You are a specialized routing LLM. Your task is to classify user queries "
            "into one of these domains: medical, code, math, legal, general.\n"
            "You must also rewrite the prompt to be optimized for a domain specialist.\n\n"
            "Domain rules:\n"
            "- code: any query asking to write, implement, or explain code or algorithms, "
            "regardless of programming language or data structure name.\n"
            "- math: equations, integrals, derivatives, proofs, numerical problems.\n"
            "- medical: health, disease, symptoms, diagnosis, treatment.\n"
            "- legal: law, rights, contracts, lawsuits, legal procedures.\n"
            "- general: philosophy, history, culture, concepts — only if none of the above fit.\n\n"
            "You MUST respond ONLY with a JSON object in this exact format:\n"
            "{\n"
            "  \"domain\": \"string\",\n"
            "  \"confidence\": float,\n"
            "  \"reasoning\": \"string\",\n"
            "  \"rewritten_prompt\": \"string\"\n"
            "}"
        )

    def classify(self, model, tokenizer=None, user_prompt: str = "", device: str = "cuda") -> dict:
        """
        Classifies the user prompt using the provided Llama model.
        tokenizer and device are kept for backward compatibility but are unused.
        """
        try:
            response_obj = model.create_chat_completion(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=256,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            response = response_obj["choices"][0]["message"]["content"]

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
