class PromptOptimizer:
    """
    Rule-based fallback optimizer for prompt rewriting when the LLM router fails.
    """
    
    TEMPLATES = {
        "medical": "You are a knowledgeable medical assistant. Answer clearly, cite general medical knowledge, and always recommend consulting a licensed physician. Question: ",
        "code": "You are an expert software engineer. Provide clean, well-commented code with explanations. Task: ",
        "math": "You are a mathematics expert. Show step-by-step working clearly. Problem: ",
        "legal": "You are a legal information assistant. Provide general legal information only, not legal advice, and recommend consulting a licensed attorney. Question: ",
        "general": ""
    }

    def optimize(self, domain: str, raw_prompt: str) -> str:
        """
        Applies a static template based on the domain.
        """
        prefix = self.TEMPLATES.get(domain, "")
        return f"{prefix}{raw_prompt}"
