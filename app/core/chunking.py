
from langchain_text_splitters import RecursiveCharacterTextSplitter
class TextChunker:
    def __init__(self, chunk_size=2000, chunk_overlap=150):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk(self, text: str):
        return self.splitter.create_documents([text])
