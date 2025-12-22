

from app.core.ingestion.sources import FileSource, URLSource
from app.core.chunking import TextChunker
from app.core.vectorstore.stores import VectorStoreFactory
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.core.logging.logger import LoggedEmbeddings

class RAGPipeline:
  
    def __init__(self, settings):
        self.embeddings = LoggedEmbeddings(GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=settings.rag.google_api_key,
             batch_size=16 
        ), settings)
        self.chunker = TextChunker()
        self.vector_factory = VectorStoreFactory(self.embeddings)
        self.settings = settings
        self.logger = settings.logger

    # * makes file and url keyword-only arguments.
    async def ingest(self, *, file=None, url=None):
        if not file and not url:
            raise ValueError("Either file or url must be provided")

        source = FileSource(file) if file else URLSource(url)

        text = await source.load()
        documents = self.chunker.chunk(text)
        self.logger.info(
            "Chunking completed",
            extra={"chunks": len(documents)}
        )

        vectorstore = self.vector_factory.create(self.settings.rag.store_type)
        self.logger.info("Starting embedding documents...")
        ids = vectorstore.add_documents(documents)
        self.logger.info("Documents added to vector store", extra={"store_type": self.settings.rag.store_type})

        return ids
