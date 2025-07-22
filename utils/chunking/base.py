from abc import ABC, abstractmethod
from typing import List

class ChunkingStrategy(ABC):
    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        pass