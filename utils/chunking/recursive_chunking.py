from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from utils.chunking.base import ChunkingStrategy


class RecursiveChunker(ChunkingStrategy):
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""]
        )

    def chunk(self, text: str) -> List[str]:
        return self.splitter.split_text(text)