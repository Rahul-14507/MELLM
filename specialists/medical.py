from .base_specialist import BaseSpecialist
from typing import Generator

class MedicalSpecialist(BaseSpecialist):
    def _get_temperature(self) -> float:
        return 0.7

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

    def stream_generate(self, prompt: str) -> Generator[str, None, None]:
        stream = self.model.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful and knowledgeable medical specialist assistant. Provide accurate and detailed information based on the user's query."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=self.max_new_tokens,
            temperature=0.7,
            stream=True
        )
        for chunk in stream:
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            token = delta.get("content", "")
            if token:
                yield token
