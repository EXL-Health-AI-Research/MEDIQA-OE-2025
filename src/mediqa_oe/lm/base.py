from abc import ABC, abstractmethod
from typing import Dict, Iterator, List

class BaseOrderExtractionLM(ABC):
    @abstractmethod
    def infer(self, messages: List, max_new_tokens: int = 2048) -> str | None:
        pass

    def get_device_info(self) -> str:
        return "N/A"

    def token_count(self, messages: List) -> int:
        raise NotImplementedError("Token counting not supported for this backend.")
    
    def infer_stream(self, messages: List, max_new_tokens: int = 2048) -> Iterator[str]:
        raise NotImplementedError("Streaming not supported for this backend.")
