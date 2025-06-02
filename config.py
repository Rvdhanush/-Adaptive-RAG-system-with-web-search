from pydantic_settings import BaseSettings
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # API Keys
    TAVILY_API_KEY: str = "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"
    
    # Model Settings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL: str = "google/flan-t5-base"  # Using a free model from HuggingFace
    USE_CUDA: bool = True  # Use GPU if available
    
    # RAG Settings
    SIMILARITY_THRESHOLD: float = 0.75  # Increased threshold for stricter matching
    MAX_LOCAL_RESULTS: int = 3  # Keep focused results
    MAX_WEB_RESULTS: int = 2  # Reduced web results
    
    # Document Processing
    CHUNK_SIZE: int = 200  # Even smaller chunks for more precise matching
    CHUNK_OVERLAP: int = 30  # Smaller overlap
    
    # Relevance Scoring
    MIN_TERM_FREQUENCY: int = 2  # Require more matching terms
    POSITION_WEIGHT: float = 0.4  # Increased weight for position
    MIN_CONTENT_LENGTH: int = 50  # Minimum content length to consider
    
    # Search Settings
    FORCE_LOCAL_SEARCH: bool = True  # Force local search first
    WEB_SEARCH_FALLBACK: bool = False  # Disable web search fallback
    
    class Config:
        env_file = ".env"

settings = Settings() 