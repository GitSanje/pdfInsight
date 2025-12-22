from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma


from .base import VectorStore


class FAISSStore(VectorStore):
    def __init__(self, embeddings):
        self.store = None
        self.embeddings = embeddings

    def add_documents(self, documents):
        self.store = FAISS.from_documents(documents, self.embeddings)
        return self

    def similarity_search(self, query, k=4):
        return self.store.similarity_search(query, k)
    
class ChromaStore(VectorStore):
    def __init__(self, embeddings, persist_dir="./chroma"):
        self.store = Chroma(
            embedding_function=embeddings,
            persist_directory=persist_dir
        )

    def add_documents(self, documents):
        self.store.add_documents(documents)
        self.store.persist()
        return self

    def similarity_search(self, query, k=4):
        return self.store.similarity_search(query, k)

class MemoryStore():
    def __init__(self, embeddings):
        self.store = InMemoryVectorStore(embedding=embeddings)

    def add_documents(self, documents):
        return self.store.add_documents(documents)

    def similarity_search(self, query, k=4):
        return self.store.similarity_search(query, k)


class VectorStoreFactory:
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def create(self, store_type: str):
        if store_type == "faiss":
            return FAISSStore(self.embeddings)
        elif store_type == "chroma":
            return ChromaStore(self.embeddings)
        elif store_type == "memory":
            return MemoryStore(self.embeddings)
        else:
            raise ValueError(f"Unsupported vector store: {store_type}")