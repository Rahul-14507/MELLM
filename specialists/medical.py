from .base_specialist import BaseSpecialist


class MedicalSpecialist(BaseSpecialist):
    def generate(self, prompt: str) -> str:
        response = self.model.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful and knowledgeable medical specialist assistant. Provide accurate and detailed information based on the user's query."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=self.max_new_tokens,
            temperature=0.7,
        )
        return self._postprocess(
            response["choices"][0]["message"]["content"]
        )
