from abc import ABC, abstractmethod

class LLMInterface(ABC):
    @abstractmethod
    def summarize(self, text: str) -> str:
        pass
