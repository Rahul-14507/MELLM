import torch
from .base_specialist import BaseSpecialist

class MedicalSpecialist(BaseSpecialist):
    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove prompt from response if model prepends it
        if response.startswith(prompt):
            response = response[len(prompt):]
            
        return self._postprocess(response)
