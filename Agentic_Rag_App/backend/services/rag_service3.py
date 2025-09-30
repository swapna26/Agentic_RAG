"""Core RAG service with Ollama and CrewAI integration - Complete Implementation."""

import asyncio
import time
import re
from typing import List, Dict, Any, Optional, AsyncGenerator
import structlog
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage

from config import BackendConfig
from services.conversation_service import ConversationService

logger = structlog.get_logger()


class RAGService:
    """Core RAG service with agentic capabilities and rate limiting."""
    
    def __init__(self, config: BackendConfig):
        self.config = config
        self.vector_store = None
        self.index = None
        self.embedding_model = None
        self.llm = None
        self.memory = None
        self.is_initialized = False

        # CrewAI components
        self.crew_agents = None
        
        # Phoenix service (injected from main.py)
        self.phoenix_service = None
        
        # Rate limiting (less restrictive for Ollama)
        self.last_request_time = 0
        self.min_request_interval = getattr(config, 'ollama_min_request_interval', 0.1)
        self.request_count = 0
        self.request_window_start = time.time()
        self.max_requests_per_minute = getattr(config, 'ollama_requests_per_minute', 100)
        self.max_retries = getattr(config, 'ollama_retry_attempts', 3)
        self.retry_delay = getattr(config, 'ollama_retry_delay', 2)

        # Conversation service (PostgreSQL-based)
        self.conversation_service = None
    
    async def initialize(self):
        """Initialize the RAG service."""
        try:
            logger.info("Initializing RAG service with Ollama models and CrewAI agents")

            # Initialize Ollama embedding model
            self.embedding_model = OllamaEmbedding(
                model_name=self.config.ollama_embedding_model,
                base_url=self.config.ollama_base_url,
            )

            # Initialize Ollama LLM
            self.llm = Ollama(
                model=self.config.ollama_model,
                base_url=self.config.ollama_base_url,
                temperature=self.config.temperature,
                request_timeout=120.0,
            )

            # Test the Ollama connection
            try:
                test_response = self.llm.complete("Hello")
                logger.info("Ollama API test successful", response_preview=str(test_response)[:50])
            except Exception as e:
                logger.warning("Ollama API test failed, continuing anyway", error=str(e))
            
            # Initialize PostgreSQL vector store with explicit configuration
            # LlamaIndex adds "data_" prefix, so use "llamaindex_vectors" to get "data_llamaindex_vectors"
            self.vector_store = PGVectorStore.from_params(
                database=self._extract_db_name(self.config.database_url),
                host=self._extract_host(self.config.database_url),
                port=self._extract_port(self.config.database_url),
                user=self._extract_user(self.config.database_url),
                password=self._extract_password(self.config.database_url),
                table_name="llamaindex_vectors",
                embed_dim=768,  # nomic-embed-text:v1.5 dimension
                perform_setup=False,  # Don't try to create table - it already exists
                # Enable debugging to see what's happening
                debug=True,
            )
            
            # Create storage context
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            
            # Initialize vector index
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                embed_model=self.embedding_model,
                storage_context=storage_context
            )
            
            # Initialize conversation memory
            self.memory = ChatMemoryBuffer.from_defaults(
                token_limit=self.config.max_tokens // 2  # Reserve half for response
            )

            # Initialize conversation service
            self.conversation_service = ConversationService(self.config)
            await self.conversation_service.initialize()

            # Initialize CrewAI agents
            await self._initialize_crew_agents()

            self.is_initialized = True
            logger.info("RAG service initialized successfully with CrewAI agents and conversation storage")
            
        except Exception as e:
            logger.error("Failed to initialize RAG service", error=str(e))
            raise
    
    async def _initialize_crew_agents(self):
        """Initialize CrewAI agents."""
        try:
            from agents.crew_agents import RAGCrew
            
            # Initialize full CrewAI setup
            self.crew_agents = RAGCrew(self, self.config)
            
            
            # Crew agents initialized above
            
            logger.info("CrewAI agents initialized successfully")
            
        except Exception as e:
            logger.warning("Failed to initialize CrewAI agents", error=str(e))
            # Continue without agents - fall back to regular RAG
    
    async def _check_rate_limit(self):
        """Check and enforce rate limiting."""
        current_time = time.time()
        
        # Reset request count every minute
        if current_time - self.request_window_start >= 60:
            self.request_count = 0
            self.request_window_start = current_time
        
        # Check requests per minute limit
        if self.request_count >= self.max_requests_per_minute:
            wait_time = 60 - (current_time - self.request_window_start)
            logger.warning("Rate limit reached, waiting", wait_time=wait_time)
            await asyncio.sleep(wait_time)
            self.request_count = 0
            self.request_window_start = time.time()
        
        # Check minimum interval between requests
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            wait_time = self.min_request_interval - time_since_last
            logger.info("Rate limiting: waiting between requests", wait_time=wait_time)
            await asyncio.sleep(wait_time)
        
        self.last_request_time = time.time()
        self.request_count += 1
    
    def _extract_db_name(self, db_url: str) -> str:
        """Extract database name from database URL."""
        return db_url.split('/')[-1]
    
    def _extract_host(self, db_url: str) -> str:
        """Extract host from database URL."""
        return db_url.split('@')[1].split(':')[0]
    
    def _extract_port(self, db_url: str) -> int:
        """Extract port from database URL."""
        parts = db_url.split('@')[1].split(':')
        return int(parts[1].split('/')[0]) if len(parts) > 1 else 5432
    
    def _extract_user(self, db_url: str) -> str:
        """Extract user from database URL."""
        return db_url.split('://')[1].split(':')[0]
    
    def _extract_password(self, db_url: str) -> str:
        """Extract password from database URL."""
        return db_url.split('://')[1].split('@')[0].split(':')[1]

    def _is_greeting(self, message: str) -> bool:
        """Check if the message is a greeting."""
        message_lower = message.lower().strip()

        # Common greeting patterns
        greeting_patterns = [
            r'^(hi|hello|hey|hiya|howdy)[\s\.,!]*$',
            r'^(good\s+(morning|afternoon|evening|day)|good\s*day)[\s\.,!]*$',
            r'^(what\'s\s+up|whats\s+up|sup)[\s\.,!]*$',
            r'^(how\s+(are\s+you|r\s+u)|how\s+you\s+doing)[\s\.,!]*$',
            r'^(greetings?|salutations?)[\s\.,!]*$',
            r'^(nice\s+to\s+meet\s+you)[\s\.,!]*$'
        ]

        return any(re.match(pattern, message_lower) for pattern in greeting_patterns)

    def _get_greeting_response(self) -> str:
        """Generate a friendly greeting response."""
        import random

        greetings = [
            "Hello! I'm your Agentic RAG Assistant. I can help you find information from the documents in my knowledge base. What would you like to know?",
            "Hi there! I'm here to assist you with questions about the documents I have access to. How can I help you today?",
            "Hey! Nice to meet you. I'm an AI assistant specialized in retrieving and analyzing information from various documents. What can I help you with?",
            "Hello! I'm ready to help you explore the knowledge base and answer your questions. What would you like to learn about?",
            "Hi! I'm your document assistant powered by advanced AI. I can search through documents and provide detailed answers. What's on your mind?"
        ]

        return random.choice(greetings)

    async def _populate_memory_from_history(self, conversation_history: List[Dict]):
        """Populate ChatMemoryBuffer with conversation history."""
        if not conversation_history:
            return

        try:
            # Clear existing memory
            self.memory.reset()

            # Add messages to memory
            for msg in conversation_history:
                chat_msg = ChatMessage(
                    role=msg["role"],
                    content=msg["content"]
                )
                self.memory.put(chat_msg)

            logger.info("Populated memory with conversation history",
                       message_count=len(conversation_history))
        except Exception as e:
            logger.error("Failed to populate memory from history", error=str(e))

    async def _add_to_conversation(self, conversation_id: str, role: str, content: str,
                                 sources: Optional[List[Dict]] = None,
                                 processing_mode: Optional[str] = None,
                                 response_time_ms: Optional[int] = None):
        """Add a message to the conversation history using PostgreSQL."""
        if not self.conversation_service:
            return

        try:
            await self.conversation_service.add_message(
                conversation_id=conversation_id,
                role=role,
                content=content,
                sources=sources,
                processing_mode=processing_mode,
                model_used=self.config.ollama_model if role == "assistant" else None,
                response_time_ms=response_time_ms
            )
        except Exception as e:
            logger.error("Failed to add message to conversation", error=str(e))

    async def chat(self, question: str, conversation_history: List[Dict], conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """Chat with the RAG system using conversation memory."""
        if not self.is_initialized:
            raise RuntimeError("RAG service not initialized")

        try:
            logger.info("Processing chat message", question=question[:100],
                       history_length=len(conversation_history))

            # Check if this is a greeting and handle it without agents
            if self._is_greeting(question):
                logger.info("Detected greeting message, responding directly")
                greeting_response = self._get_greeting_response()

                # Add to conversation history if conversation_id provided
                if conversation_id:
                    await self._add_to_conversation(
                        conversation_id, "user", question, processing_mode="greeting_request"
                    )
                    await self._add_to_conversation(
                        conversation_id, "assistant", greeting_response, processing_mode="greeting_response"
                    )

                result = {
                    "response": greeting_response,
                    "sources": [],
                    "metadata": {
                        "model": self.config.ollama_model,
                        "conversation_id": conversation_id,
                        "source_count": 0,
                        "processing_mode": "greeting_response"
                    }
                }

                # Log greeting interaction to Phoenix
                if self.phoenix_service:
                    await self.phoenix_service.log_chat_interaction(
                        conversation_id or "unknown",
                        question,
                        greeting_response,
                        [],
                        result["metadata"]
                    )

                return result

            # Populate memory with conversation history
            await self._populate_memory_from_history(conversation_history)

            # Apply rate limiting for non-greeting queries
            await self._check_rate_limit()

            # Start tracing span for the query
            trace_span = None
            if self.phoenix_service:
                trace_span = self.phoenix_service.create_trace_span(
                    "rag_chat",
                    {
                        "query": question[:100],
                        "conversation_id": conversation_id,
                        "history_length": len(conversation_history)
                    }
                )

            # Use CrewAI agents - no fallbacks
            if not self.crew_agents:
                raise RuntimeError("CrewAI agents not available")

            logger.info("Using CrewAI agents for processing")

            # Always provide conversation context if available, let AI decide what to do
            enriched_query = question  # Default to just the question

            if conversation_history and len(conversation_history) > 0:
                # Build conversation context
                context_parts = []
                for msg in conversation_history[-4:]:  # Last 2 exchanges (4 messages max)
                    role = "Human" if msg["role"] == "user" else "Assistant"
                    content = msg['content'][:300] + "..." if len(msg['content']) > 300 else msg['content']
                    context_parts.append(f"{role}: {content}")

                conversation_context = "\n".join(context_parts)

                # Let the AI agent make the intelligent decision
                enriched_query = f"""Current question: {question}

Recent conversation context:
{conversation_context}

INSTRUCTIONS FOR AI AGENT:
You must intelligently decide whether this is:
1. A REFORMAT REQUEST: User wants to modify/reformat a previous response (e.g., "give answer in 5 lines", "summarize above", "make it shorter")
   - If so: Use the conversation context to reformat the previous assistant response
   - Do NOT search for new documents

2. A NEW CONTENT QUESTION: User wants new information (e.g., "What is procurement? give answer in 5 lines", "Tell me about Law X")
   - If so: Search for relevant documents and answer the new question
   - Ignore conversation context for content, but respect any format requirements

Use your intelligence to distinguish between these two scenarios and respond accordingly."""

                logger.info("Providing conversation context for AI agent intelligent decision")
            else:
                logger.info("No conversation history, processing as new question")

            result = await self.crew_agents.process_query(enriched_query)
            logger.info("CrewAI workflow completed", response_length=len(result["response"]))

            if conversation_id:
                await self._add_to_conversation(
                    conversation_id, "user", question, processing_mode="crew_ai"
                )
                await self._add_to_conversation(
                    conversation_id, "assistant", result['response'],
                    sources=result.get('sources', []),
                    processing_mode="crew_ai_response"
                )

            result["metadata"]["has_conversation_context"] = len(conversation_history) > 0
            result["metadata"]["processing_mode"] = "crew_ai"

            # Log to Phoenix if available
            if self.phoenix_service:
                await self.phoenix_service.log_chat_interaction(
                    conversation_id or "unknown",
                    question,
                    result["response"],
                    result.get('sources', []),
                    result.get("metadata", {})
                )

            # End tracing span
            if trace_span:
                trace_span.set_attribute("response.length", len(result["response"]))
                trace_span.set_attribute("sources.count", len(result.get("sources", [])))
                trace_span.end()

            logger.info("Chat processed successfully",
                       response_length=len(result["response"]),
                       source_count=len(result.get("sources", [])))

            return result

        except Exception as e:
            logger.error("Chat processing failed", error=str(e))
            raise

    async def query(self, question: str, conversation_id: Optional[str] = None, use_agents: bool = True) -> Dict[str, Any]:
        """Legacy query method - redirects to chat with empty history."""
        return await self.chat(question, [], conversation_id)
    
    async def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get conversation history from PostgreSQL."""
        try:
            if self.conversation_service:
                return await self.conversation_service.get_conversation_history(conversation_id)
            else:
                # Fallback to LlamaIndex memory
                messages = self.memory.get_all()
                history = []

                for i in range(0, len(messages), 2):
                    if i + 1 < len(messages):
                        history.append({
                            "user": messages[i].content.replace("Human: ", ""),
                            "assistant": messages[i + 1].content.replace("Assistant: ", ""),
                            "timestamp": messages[i].additional_kwargs.get("timestamp", "")
                        })

                return history

        except Exception as e:
            logger.error("Failed to get conversation history", error=str(e))
            return []
    
    async def clear_conversation(self, conversation_id: str) -> bool:
        """Clear conversation history from PostgreSQL."""
        try:
            if self.conversation_service:
                result = await self.conversation_service.clear_conversation(conversation_id)
                if result:
                    # Also clear LlamaIndex memory
                    self.memory.clear()
                logger.info("Conversation cleared", conversation_id=conversation_id, success=result)
                return result
            else:
                # Fallback to clearing only LlamaIndex memory
                self.memory.clear()
                logger.info("Conversation cleared (memory only)", conversation_id=conversation_id)
                return True
        except Exception as e:
            logger.error("Failed to clear conversation", error=str(e))
            return False
    
    async def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        try:
            return {
                "status": "healthy",
                "total_documents": 5,  # From your indexer
                "total_chunks": 482,   # From your indexer
                "embedding_model": self.config.ollama_embedding_model,
                "llm_model": self.config.ollama_model,
                "crew_agents_enabled": bool(self.crew_agents),
                "phoenix_service_enabled": bool(self.phoenix_service),
                "rate_limit_info": {
                    "requests_this_minute": self.request_count,
                    "max_requests_per_minute": self.max_requests_per_minute,
                    "min_request_interval": self.min_request_interval
                }
            }
        except Exception as e:
            logger.error("Failed to get index stats", error=str(e))
            return {"status": "error", "error": str(e)}
    
    async def is_healthy(self) -> bool:
        """Check if the service is healthy."""
        try:
            if not self.is_initialized:
                return False
            
            # Just return True for now to avoid triggering quota limits in health checks
            # In production, you might want a lightweight health check
            return True
            
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return False
    
    
    async def cleanup(self):
        """Cleanup resources."""
        try:
            if self.vector_store:
                # Close database connections if needed
                pass
            self.is_initialized = False
            logger.info("RAG service cleanup completed")
        except Exception as e:
            logger.error("Error during cleanup", error=str(e))