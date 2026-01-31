"""Configuration management for the RAG Chat Assistant."""

import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # OpenAI Configuration
    openai_api_key: str
    openai_model: str = "gpt-4-turbo-preview"
    openai_embedding_model: str = "text-embedding-3-small"

    # Chroma Configuration
    chroma_db_path: str = "./data/chroma_db"
    chroma_collection_name: str = "documents"

    # MCP Server Configuration
    mcp_server_transport: str = "stdio"
    mcp_server_command: str = "python"
    mcp_server_args: str = "-m src.mcp_server.server"

    # Application Configuration
    streamlit_port: int = 8501
    streamlit_host: str = "0.0.0.0"
    log_level: str = "INFO"

    # Chunking Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrieval Configuration
    top_k_chunks: int = 5
    relevance_threshold: float = 0.3  # Lowered from 0.7 to avoid filtering out all results

    # ReAct Agent Configuration
    max_iterations: int = 10
    temperature: float = 0.7

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def chroma_db_path_resolved(self) -> Path:
        """Get resolved Chroma DB path."""
        return Path(self.chroma_db_path).resolve()

    def ensure_data_directories(self) -> None:
        """Ensure data directories exist."""
        self.chroma_db_path_resolved.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()

# Ensure data directories exist on import
settings.ensure_data_directories()
