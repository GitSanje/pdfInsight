from pydantic_settings import BaseSettings, SettingsConfigDict

class BaseAppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file="./.env",
        extra="ignore",
        case_sensitive=False,
    )