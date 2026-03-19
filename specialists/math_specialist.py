import torch
from .base_specialist import BaseSpecialist

class MathSpecialist(BaseSpecialist):
    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=min(self.max_new_tokens, 300),
                use_cache=False
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if response.startswith(prompt):
            response = response[len(prompt):]
            
        return self._postprocess(response)
