
from .base import DocumentParser
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from fastapi import UploadFile

class PDFParser(DocumentParser):
    async def parse(self, content: bytes) -> str:
        with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
            f.write(content)
            f.flush()
            loader = PyPDFLoader(f.name)
            docs = loader.load_and_split()
        return "\n".join(d.page_content for d in docs)

class TextParser(DocumentParser):
    async def parse(self, content: bytes) -> str:
        return content.decode("utf-8", errors="ignore")


class ParserFactory:
    @staticmethod
    def get_parser_file(file: UploadFile) -> DocumentParser:
        if file.content_type == "application/pdf":
            return PDFParser()
        if file.filename.endswith((".txt", ".csv")):
            return TextParser()
        raise ValueError("Unsupported file type")

    @staticmethod
    def get_parser_url(content_type: str) -> DocumentParser:
        if "application/pdf" in content_type:
            return PDFParser()
        if "text/plain" in content_type or "text/csv" in content_type:
            return TextParser()
        raise ValueError("Unsupported content type")
