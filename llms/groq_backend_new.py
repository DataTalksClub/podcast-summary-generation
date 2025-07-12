import os
from typing import Optional
from dotenv import load_dotenv
from groq import Groq

from llms.base import LLMInterface  # Remove if unused
#from pipeline.prompt_template_new import build_prompt
#from pipeline.prompt_chunking import build_prompt
from pipeline.prompt_chunking_v2 import build_prompt

load_dotenv()

class GroqLLM(LLMInterface):
    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment.")
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"
        #self.model = "llama-3.1-8b-instant"

    def summarize(self, text: str, format_type: Optional[str] = None) -> str:
        """
        Summarizes or extracts content from text using the Groq LLM.

        Args:
            text: The transcript or text chunk to summarize.
            format_type: Optional; type of prompt to use.
                         If None, uses combined prompt covering all points.

        Returns:
            The LLM response string.
        """
        #prompt = build_prompt(text, format_type=format_type)
        prompt = build_prompt(text) 
        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model
        )
        return response.choices[0].message.content.strip()
