from .base_specialist import BaseSpecialist


class MathSpecialist(BaseSpecialist):
    def generate(self, prompt: str) -> str:
        response = self.model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=min(self.max_new_tokens, 512),
            temperature=0.1,
        )
        return self._postprocess(
            response["choices"][0]["message"]["content"]
        )
