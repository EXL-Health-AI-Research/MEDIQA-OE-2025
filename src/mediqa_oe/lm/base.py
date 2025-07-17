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
    
# Agent Classes
class BaseAgent(ABC):
    """Base class for all agents"""

    def __init__(self, llm: OrderExtractionLM):
        self.llm = llm

    @abstractmethod
    def process(self, *args, **kwargs) -> Any:
        pass

    def _create_messages(self, system_prompt: str, user_input: str) -> List[Dict[str, str]]:
        """Create formatted messages for the medical model"""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
