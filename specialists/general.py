from .base_specialist import BaseSpecialist


class GeneralSpecialist(BaseSpecialist):
    def generate(self, prompt: str) -> str:
        response = self.model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_new_tokens,
            temperature=0.7,
        )
        return self._postprocess(
            response["choices"][0]["message"]["content"]
        )
