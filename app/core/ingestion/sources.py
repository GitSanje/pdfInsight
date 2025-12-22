
import httpx
from fastapi import UploadFile
from app.core.parsers.parsers import ParserFactory
from .base import BaseSource

class FileSource(BaseSource):
    def __init__(self, file: UploadFile):
        self.file = file

    async def load(self) -> str:
        content = await self.file.read()
        parser = ParserFactory.get_parser_file(self.file)
        return await parser.parse(content)
 

class URLSource(BaseSource):
    def __init__(self, url: str):
        self.url = url
    async def load(self) -> str:
        if self.url.startswith("http://") or self.url.startswith("https://"):
           async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(self.url)
            response.raise_for_status()

            parser = ParserFactory.get_parser_url(
                response.headers.get("content-type", "")
            )
            return await parser.parse(response.content)
        raise ValueError("Unsupported url")