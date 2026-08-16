"""
Configuration Management for ISMS RAG Platform
==============================================

This module provides centralized configuration management for the ISMS RAG platform.
It loads environment variables from a .env file and provides type-safe access to
all configuration settings.

Usage:
    from config import Config
    
    # Access configuration
    api_key = Config.OPENAI_API_KEY
    model = Config.LLM_MODEL
    
    # Validate configuration before use
    Config.validate()

Date: 2026
"""

import os
import sys
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
# Searches for .env in current directory and parent directories
load_dotenv()


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing required values."""
    pass


class Config:
    """
    Central configuration class for ISMS RAG Platform.
    
    All configuration values are loaded from environment variables with sensible
    defaults where applicable. Required values without defaults will cause
    validation errors if not provided.
    
    Attributes:
        BASE_DIR: Project root directory
        DATA_DIR: Directory for data files
        FRAMEWORKS_DIR: Directory for framework documents
        POLICIES_DIR: Directory for policy documents
        OPENAI_API_KEY: OpenAI API key (required)
        LLM_MODEL: OpenAI model name for text generation
        LLM_TEMPERATURE: Temperature parameter for LLM (0.0-1.0)
        MAX_TOKENS: Maximum tokens in LLM responses
        EMBEDDING_MODEL: Model name for text embeddings
        CHROMA_PERSIST_DIRECTORY: ChromaDB persistence directory
        CHROMA_COLLECTION_NAME: ChromaDB collection name
        CHUNK_SIZE: Size of text chunks for document splitting
        CHUNK_OVERLAP: Overlap between consecutive chunks
        DEFAULT_LANGUAGE: Default response language (fr/en)
        SUPPORTED_LANGUAGES: List of supported languages
        TOP_K_RESULTS: Number of results to retrieve
        SIMILARITY_THRESHOLD: Minimum similarity score for retrieval
        APP_ENV: Application environment (development/staging/production)
        LOG_LEVEL: Logging level
        DEBUG: Debug mode flag
    """
    
    # =========================================================================
    # Project Paths
    # =========================================================================
    
    BASE_DIR: Path = Path(__file__).parent.parent
    """Project root directory"""
    
    DATA_DIR: Path = BASE_DIR / "data"
    """Directory containing all data files"""
    
    FRAMEWORKS_DIR: Path = DATA_DIR / "frameworks"
    """Directory for security framework documents (ISO 27001, NIST, etc.)"""
    
    POLICIES_DIR: Path = DATA_DIR / "policies"
    """Directory for organizational policy documents"""
    
    # =========================================================================
    # API Keys
    # =========================================================================
    
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    """OpenAI API key - Required for LLM and embeddings"""
    
    # =========================================================================
    # Language Model (LLM) Settings
    # =========================================================================
    
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4")
    """OpenAI model for text generation (gpt-4, gpt-4-turbo, gpt-3.5-turbo)"""
    
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    """Temperature for LLM responses (0.0 = deterministic, 1.0 = creative)"""
    
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "2000"))
    """Maximum number of tokens in LLM responses"""
    
    # =========================================================================
    # Embedding Model Settings
    # =========================================================================
    
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    """Model for generating text embeddings (supports multiple languages)"""
    
    # =========================================================================
    # Vector Database Settings
    # =========================================================================
    
    CHROMA_PERSIST_DIRECTORY: str = os.getenv(
        "CHROMA_PERSIST_DIRECTORY",
        "./chroma_db"
    )
    """Directory for ChromaDB persistence"""
    
    CHROMA_COLLECTION_NAME: str = os.getenv(
        "CHROMA_COLLECTION_NAME",
        "isms_frameworks"
    )
    """ChromaDB collection name for document embeddings"""
    
    # =========================================================================
    # Document Processing Settings
    # =========================================================================
    
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    """Size of text chunks for document splitting (in characters)"""
    
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    """Number of overlapping characters between consecutive chunks"""
    
    # =========================================================================
    # Language Configuration
    # =========================================================================
    
    DEFAULT_LANGUAGE: str = os.getenv("DEFAULT_LANGUAGE", "fr")
    """Default language for responses (fr=French, en=English)"""
    
    SUPPORTED_LANGUAGES: List[str] = os.getenv(
        "SUPPORTED_LANGUAGES",
        "fr,en"
    ).split(",")
    """List of supported languages"""
    
    # =========================================================================
    # Retrieval Configuration
    # =========================================================================
    
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))
    """Number of similar documents to retrieve for each query"""
    
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    """Minimum similarity score (0.0-1.0) for retrieved documents"""
    
    # =========================================================================
    # Application Settings
    # =========================================================================
    
    APP_ENV: str = os.getenv("APP_ENV", "development")
    """Application environment (development/staging/production)"""
    
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    """Logging level (DEBUG/INFO/WARNING/ERROR/CRITICAL)"""
    
    DEBUG: bool = os.getenv("DEBUG", "true").lower() in ("true", "1", "yes")
    """Enable debug mode with detailed error messages"""
    
    # =========================================================================
    # Methods
    # =========================================================================
    
    @classmethod
    def validate(cls) -> None:
        """
        Validate configuration settings.
        
        Checks that all required configuration values are set and that
        values are within acceptable ranges.
        
        Raises:
            ConfigurationError: If any required config is missing or invalid
            
        Example:
            >>> Config.validate()  # Raises error if OPENAI_API_KEY not set
        """
        errors: List[str] = []
        
        # Check required API key
        if not cls.OPENAI_API_KEY:
            errors.append(
                "OPENAI_API_KEY is required. "
                "Get your key from https://platform.openai.com/api-keys"
            )
        
        # Validate temperature range
        if not 0.0 <= cls.LLM_TEMPERATURE <= 1.0:
            errors.append(
                f"LLM_TEMPERATURE must be between 0.0 and 1.0, "
                f"got {cls.LLM_TEMPERATURE}"
            )
        
        # Validate chunk settings
        if cls.CHUNK_SIZE <= 0:
            errors.append(f"CHUNK_SIZE must be positive, got {cls.CHUNK_SIZE}")
        
        if cls.CHUNK_OVERLAP < 0:
            errors.append(
                f"CHUNK_OVERLAP must be non-negative, got {cls.CHUNK_OVERLAP}"
            )
        
        if cls.CHUNK_OVERLAP >= cls.CHUNK_SIZE:
            errors.append(
                f"CHUNK_OVERLAP ({cls.CHUNK_OVERLAP}) must be less than "
                f"CHUNK_SIZE ({cls.CHUNK_SIZE})"
            )
        
        # Validate language settings
        if cls.DEFAULT_LANGUAGE not in cls.SUPPORTED_LANGUAGES:
            errors.append(
                f"DEFAULT_LANGUAGE '{cls.DEFAULT_LANGUAGE}' not in "
                f"SUPPORTED_LANGUAGES {cls.SUPPORTED_LANGUAGES}"
            )
        
        # Validate retrieval settings
        if cls.TOP_K_RESULTS <= 0:
            errors.append(
                f"TOP_K_RESULTS must be positive, got {cls.TOP_K_RESULTS}"
            )
        
        if not 0.0 <= cls.SIMILARITY_THRESHOLD <= 1.0:
            errors.append(
                f"SIMILARITY_THRESHOLD must be between 0.0 and 1.0, "
                f"got {cls.SIMILARITY_THRESHOLD}"
            )
        
        # Validate max tokens
        if cls.MAX_TOKENS <= 0:
            errors.append(f"MAX_TOKENS must be positive, got {cls.MAX_TOKENS}")
        
        # Validate log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if cls.LOG_LEVEL.upper() not in valid_log_levels:
            errors.append(
                f"LOG_LEVEL must be one of {valid_log_levels}, "
                f"got '{cls.LOG_LEVEL}'"
            )
        
        # If there are errors, raise with all messages
        if errors:
            error_message = "Configuration validation failed:\n" + "\n".join(
                f"  - {error}" for error in errors
            )
            raise ConfigurationError(error_message)
    
    @classmethod
    def get_data_path(cls, *paths: str) -> Path:
        """
        Get absolute path within data directory.
        
        Args:
            *paths: Path components to join with DATA_DIR
            
        Returns:
            Absolute path within data directory
            
        Example:
            >>> Config.get_data_path("frameworks", "ISO27001.pdf")
            PosixPath('/path/to/project/data/frameworks/ISO27001.pdf')
        """
        return cls.DATA_DIR.joinpath(*paths)
    
    @classmethod
    def ensure_directories(cls) -> None:
        """
        Create all required directories if they don't exist.
        
        Creates:
            - DATA_DIR
            - FRAMEWORKS_DIR
            - POLICIES_DIR
            - CHROMA_PERSIST_DIRECTORY
            
        Example:
            >>> Config.ensure_directories()
        """
        directories = [
            cls.DATA_DIR,
            cls.FRAMEWORKS_DIR,
            cls.POLICIES_DIR,
            Path(cls.CHROMA_PERSIST_DIRECTORY),
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def display_config(cls) -> None:
        """
        Display current configuration (for debugging).
        
        Hides sensitive values like API keys.
        
        Example:
            >>> Config.display_config()
            Configuration:
              LLM_MODEL: gpt-4
              EMBEDDING_MODEL: sentence-transformers/...
              ...
        """
        print("Configuration:")
        print(f"  LLM_MODEL: {cls.LLM_MODEL}")
        print(f"  LLM_TEMPERATURE: {cls.LLM_TEMPERATURE}")
        print(f"  MAX_TOKENS: {cls.MAX_TOKENS}")
        print(f"  EMBEDDING_MODEL: {cls.EMBEDDING_MODEL}")
        print(f"  CHROMA_COLLECTION_NAME: {cls.CHROMA_COLLECTION_NAME}")
        print(f"  CHUNK_SIZE: {cls.CHUNK_SIZE}")
        print(f"  CHUNK_OVERLAP: {cls.CHUNK_OVERLAP}")
        print(f"  DEFAULT_LANGUAGE: {cls.DEFAULT_LANGUAGE}")
        print(f"  SUPPORTED_LANGUAGES: {cls.SUPPORTED_LANGUAGES}")
        print(f"  TOP_K_RESULTS: {cls.TOP_K_RESULTS}")
        print(f"  SIMILARITY_THRESHOLD: {cls.SIMILARITY_THRESHOLD}")
        print(f"  APP_ENV: {cls.APP_ENV}")
        print(f"  LOG_LEVEL: {cls.LOG_LEVEL}")
        print(f"  DEBUG: {cls.DEBUG}")
        print(f"  OPENAI_API_KEY: {'*' * 10 if cls.OPENAI_API_KEY else 'NOT SET'}")


# Auto-validate configuration on import in non-development environments
if Config.APP_ENV != "development" and __name__ != "__main__":
    try:
        Config.validate()
    except ConfigurationError as e:
        print(f"Configuration Error: {e}", file=sys.stderr)
        sys.exit(1)

