
from pydantic import Field
from .base import BaseAppSettings

class RAGSettings(BaseAppSettings):
    google_api_key: str = Field(validation_alias="GOOGLE_API_KEY")
    store_type :str = "faiss"

class RedisSettings(BaseAppSettings):
    redis_url: str = "redis://localhost:6379"



class AppSettings:
    def __init__(self, logger=None):
        self.rag = RAGSettings()
        self.redis = RedisSettings()
        self.logger = logger

# print(RAGSettings().model_dump())