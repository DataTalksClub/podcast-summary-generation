import json
from typing import List, Optional
from utils.prompt import load_prompt
from utils.preprocess import preprocess_transcript
from llms.base import LLMInterface
from utils.chunking.base import ChunkingStrategy



class PodcastSummarizationPipeline:
    """
    Abstract base class for podcast summarization pipelines.
    Handles preprocessing, chunking, and prompt loading.
    Subclasses must implement `run()`.
    """

    def __init__(
        self,
        llm: LLMInterface,
        chunker: ChunkingStrategy,
        user_prompt_path: str,
        system_prompt_path: Optional[str] = None
    ):
        """
        Initialize the pipeline with model, chunking strategy, and prompt paths.

        Args:
            llm: An instance of LLMInterface for text generation.
            chunker: A strategy implementing ChunkingStrategy.
            user_prompt_path: Path to the user-facing prompt template (.txt).
            system_prompt_path: Path to the system prompt template (.txt), optional.
        """
        self.llm = llm
        self.chunker = chunker
        self.user_prompt_template = load_prompt(user_prompt_path)
        self.system_prompt = load_prompt(system_prompt_path) if system_prompt_path else None
        self._chunks = []  # stores chunks for inspection

    def run(self, raw_text: str) -> dict:
        """
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Use a subclass with specific summarization logic.")

    def preprocess_and_chunk(self, raw_text: str) -> List[str]:
        """
        Preprocess the raw transcript and chunk it using the provided strategy.

        Args:
            raw_text: The raw Markdown or transcript string.

        Returns:
            List of text chunks (strings).
        """
        clean = preprocess_transcript(raw_text)
        self._chunks = self.chunker.chunk(clean)
        return self._chunks

    def get_chunks(self) -> List[str]:
        """
        Returns the last processed chunks (after `run()`).

        Returns:
            List of text chunks.
        """
        return self._chunks

    def run_from_file(self, input_path: str, output_path: Optional[str] = None) -> dict:
        """
        Load transcript from a file, run the pipeline, and optionally save output.

        Args:
            input_path: Path to the .md or .txt transcript file.
            output_path: Optional path to save the summary as JSON.

        Returns:
            Result dict (typically includes 'summary', optional 'partials', etc.)
        """
        with open(input_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        result = self.run(raw_text)

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            print(f"✅ Summary saved to {output_path}")

        return result
