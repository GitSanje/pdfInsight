
from abc import ABC, abstractmethod

class BaseSource(ABC):
    @abstractmethod
    async def load(self) -> str:
        """Return raw text"""
        
