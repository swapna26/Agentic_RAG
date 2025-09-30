"""Configuration for the Document Indexer with Gemini."""

import os
from pathlib import Path
from dotenv import load_dotenv


class IndexerConfig:
    """Configuration class using environment variables for Gemini-based indexing."""

    def __init__(self):
        # Load environment variables
        self._load_env()

        # Database
        self.database_url = os.getenv(
            'DATABASE_URL',
            'postgresql://raguser:ragpassword@localhost:5432/agentic_rag'
        )

        # LLM Provider Selection
        self.llm_provider = os.getenv('LLM_PROVIDER', 'gemini')  # 'ollama' or 'gemini'

        # Ollama API (backup/fallback)
        self.ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        self.ollama_model = os.getenv('OLLAMA_MODEL', 'llama3.2:1b')
        self.ollama_embedding_model = os.getenv('OLLAMA_EMBEDDING_MODEL', 'nomic-embed-text:v1.5')

        # Gemini API (primary for indexing)
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.gemini_model = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')
        self.gemini_embedding_model = os.getenv('GEMINI_EMBEDDING_MODEL', 'text-embedding-004')

        # Phoenix
        self.phoenix_base_url = os.getenv('PHOENIX_BASE_URL', 'http://localhost:6006')
        self.phoenix_project_name = os.getenv('PHOENIX_PROJECT_NAME', 'agentic_rag_indexer')

        # Document processing
        self.documents_path = os.getenv('DOCUMENTS_PATH', './documents')
        self.markdown_path = os.getenv('MARKDOWN_PATH', '../markdown_output')

        # Improved chunking strategy for better text splitting
        self.chunk_size = int(os.getenv('CHUNK_SIZE', '1000'))  # Increased for better context
        self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', '200'))  # Increased overlap
        self.max_chunk_size = int(os.getenv('MAX_CHUNK_SIZE', '1500'))

        # Processing
        self.batch_size = int(os.getenv('BATCH_SIZE', '5'))  # Smaller batches for Gemini API

        # Logging
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')

        # Validate required settings
        self._validate()
    
    def _load_env(self):
        """Load environment variables from .env file."""
        env_paths = [
            Path(__file__).parent / ".env",
            Path(__file__).parent.parent / ".env",
            Path.cwd() / ".env"
        ]
        
        for env_path in env_paths:
            if env_path.exists():
                load_dotenv(env_path)
                print(f"Loaded environment from: {env_path}")
                return
        
        print("No .env file found, using defaults")
    
    def _validate(self):
        """Validate configuration."""
        if not self.database_url.startswith(('postgresql://', 'postgresql+psycopg2://')):
            raise ValueError('DATABASE_URL must be a PostgreSQL connection string')

        # Validate LLM provider specific settings
        if self.llm_provider == 'gemini':
            if not self.gemini_api_key:
                raise ValueError('GEMINI_API_KEY is required when using Gemini provider')
        elif self.llm_provider == 'ollama':
            if not self.ollama_base_url:
                print("Warning: OLLAMA_BASE_URL not set. Using default: http://localhost:11434")
        else:
            raise ValueError(f'Invalid LLM_PROVIDER: {self.llm_provider}. Must be "ollama" or "gemini"')
        
        

# Global config instance
config = IndexerConfig()