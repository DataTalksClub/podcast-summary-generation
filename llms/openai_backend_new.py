import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from llms.base import LLMInterface
#from pipeline.prompt_template import build_prompt
from pipeline.prompt_chunking_v2 import build_prompt

load_dotenv()


class OpenAILLM(LLMInterface):
    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment.")
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"  # For 'mini' usage, just use the same name; OpenAI handles scaling internally
        #self.model = "gpt-4o" 

    def summarize(self, text: str) -> str:
        prompt = build_prompt(text)
        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}], model=self.model
        )
        return response.choices[0].message.content.strip()
