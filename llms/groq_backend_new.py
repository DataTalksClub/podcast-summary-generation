import os
from typing import Optional
from dotenv import load_dotenv
from groq import Groq

from llms.base import LLMInterface  # Remove if unused
from pipeline.prompt_chunking_v2 import build_prompt

load_dotenv()

class GroqLLM(LLMInterface):
    def __init__(self, model: str = "llama-3.3-70b-versatile", api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment.")
        self.client = Groq(api_key=self.api_key)
        self.model = model 

    def generate(self, prompt:str, system_prompt:Optional[str] = None) -> str:

        messages = [] 

        if system_prompt:
            messages.append({"role":"system", "content":system_prompt})
        messages.append({"role":"user", "content":prompt})

        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model
            )
            return response.choices[0].message.content.stript()
        except Exception as e:
            raise   RuntimeError(f"Groq API call failed: {e}")


