from pydantic_settings import BaseSettings
from functools import lru_cache


# Defines all env vars the app needs - auto-loads from .env file or environment
class Settings(BaseSettings):
    # Threads API credentials (get from Meta Developer Console)
    threads_app_id: str
    threads_app_secret: str
    threads_access_token: str
    threads_verify_token: str  # random string you make up for webhook security

    openai_api_key: str  # from platform.openai.com

    database_url: str  # Neon postgres connection string

    debug: bool = False  # True = uses mock APIs for testing

    class Config:
        env_file = ".env"  # reads from .env file in project root


@lru_cache  # only loads settings once, reuses after
def get_settings() -> Settings:
    return Settings()
