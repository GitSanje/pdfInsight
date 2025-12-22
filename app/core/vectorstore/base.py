
from abc import ABC, abstractmethod

class VectorStore(ABC):
    @abstractmethod
    def add_documents(self, documents):
        pass

    @abstractmethod
    def similarity_search(self, query: str, k: int = 4):
        pass
