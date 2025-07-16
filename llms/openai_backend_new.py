import os
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI
from llms.base import LLMInterface
load_dotenv()


class OpenAILLM(LLMInterface):
    def __init__(self, model: str = "got-4o-mini", api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment.")
        self.client = OpenAI(api_key=self.api_key)
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
            raise   RuntimeError(f"OpenAI API call failed: {e}")