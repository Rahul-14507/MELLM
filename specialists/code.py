from .base_specialist import BaseSpecialist


class CodeSpecialist(BaseSpecialist):
    def generate(self, prompt: str) -> str:
        response = self.model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_new_tokens,
            temperature=0.1,
            stop=["```\n\n"]
        )
        return self._postprocess(
            response["choices"][0]["message"]["content"]
        )
