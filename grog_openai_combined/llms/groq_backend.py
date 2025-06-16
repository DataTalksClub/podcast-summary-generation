import os
from groq import Groq
from dotenv import load_dotenv
from llms.base import LLMInterface
from pipeline.prompt_template import build_prompt
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()


class GroqLLM(LLMInterface):
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment.")
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"
        #self.model = "llama-3.1-8b-instant"

    def summarize(self, text: str) -> str:
        prompt = build_prompt(text)
        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model
        )
        return response.choices[0].message.content.strip()
