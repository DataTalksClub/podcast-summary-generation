
import os
from typing import Optional
from dotenv import load_dotenv
from groq import Groq

from llms.base import LLMInterface
from pipeline.prompt_template_new import build_prompt

load_dotenv()


class GroqLLM(LLMInterface):
    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment.")
        self.client = Groq(api_key=api_key)
        self.model = "gemma2-9b-it"

    def summarize(self, text: str) -> str:
        prompt = build_prompt(text)
        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}], model=self.model
        )
        return response.choices[0].message.content.strip()

