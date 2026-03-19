from .base_specialist import BaseSpecialist


class MedicalSpecialist(BaseSpecialist):
    def generate(self, prompt: str) -> str:
        response = self.model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=min(self.max_new_tokens, 512),
            temperature=0.7,
        )
        return self._postprocess(
            response["choices"][0]["message"]["content"]
        )
