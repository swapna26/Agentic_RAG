"""
RAG Service with CrewAI Integration

Core RAG service for document retrieval and response generation using CrewAI agents.
Integrates with PostgreSQL vector store and manages conversation context.
"""

import asyncio
import time
import re
from typing import List, Dict, Any, Optional
import structlog
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
# Removed ChatMemoryBuffer and ChatMessage - using PostgreSQL only for conversation storage

from config import BackendConfig
from services.conversation_service import ConversationService

# Removed guardrails imports - keeping system simple

logger = structlog.get_logger()


class RAGService:
    """
    RAG Service with CrewAI Integration

    Provides document retrieval and response generation using CrewAI agents.
    Integrates with PostgreSQL vector store and manages conversation context.
    """
    
    def __init__(self, config: BackendConfig):
        """Initialize the RAG service with configuration."""
        self.config = config
        
        # Core components
        self.vector_store = None
        self.index = None
        self.embedding_model = None
        self.llm = None
        self.is_initialized = False

        # CrewAI agents
        self.crew_agents = None

        # External services
        self.phoenix_service = None
        self.conversation_service = None

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.5
        self.request_count = 0
        self.request_window_start = time.time()
        self.max_requests_per_minute = 30
        self.max_retries = 2
        self.retry_delay = 1

        # Backup query engine (if needed)
        # self.query_engine = None

    
    async def initialize(self):
        """Initialize the RAG service components."""
        try:
            logger.info("Initializing Agentic RAG service with CrewAI integration")

            # Initialize Ollama embedding model
            self.embedding_model = OllamaEmbedding(
                model_name=self.config.ollama_embedding_model,
                base_url=self.config.ollama_base_url,
            )

            # Initialize Ollama LLM with standardized timeout
            self.llm = Ollama(
                model=self.config.ollama_model,
                base_url=self.config.ollama_base_url,
                temperature=self.config.temperature,
                request_timeout=60.0,  # Standardized to 60s across all components
            )

            # Test the Ollama connection
            try:
                test_response = self.llm.complete("Hello")
                logger.info("Ollama API test successful", response_preview=str(test_response)[:50])
            except Exception as e:
                logger.warning("Ollama API test failed, continuing anyway", error=str(e))
            
            # Initialize PostgreSQL vector store with explicit configuration
            # LlamaIndex adds "data_" prefix, so use "llamaindex_vectors_copy" to get "data_llamaindex_vectors_copy"
            self.vector_store = PGVectorStore.from_params(
                database=self._extract_db_name(self.config.database_url),
                host=self._extract_host(self.config.database_url),
                port=self._extract_port(self.config.database_url),
                user=self._extract_user(self.config.database_url),
                password=self._extract_password(self.config.database_url),
                table_name="llamaindex_vectors_copy",
                embed_dim=768,  # Match the actual database embedding dimensions
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
            
            # Initialize query engine with retrieval settings
            self.query_engine = self.index.as_query_engine(
                llm=self.llm,
                similarity_top_k=self.config.similarity_top_k,
                response_mode="compact",
                verbose=True
            )
            
            # Remove LlamaIndex memory - we'll use PostgreSQL only for conversation storage
            self.memory = None

            # Initialize chat engine without memory (PostgreSQL handles conversation context)
            try:
                self.chat_engine = self.index.as_chat_engine(
                    chat_mode="simple",  # Use simple mode without memory
                    llm=self.llm,
                    verbose=True
                )
                logger.info("Chat engine initialized successfully (using PostgreSQL for conversation memory)")
            except Exception as e:
                logger.warning("Failed to initialize chat engine", error=str(e))
                self.chat_engine = None

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

    # Topic change detection removed - now handled at router level for better consistency

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
        """Legacy method - now uses PostgreSQL only for conversation storage."""
        # Memory management is now handled entirely by PostgreSQL conversation service
        # This method is kept for compatibility but does nothing
        if conversation_history:
            logger.info("Using PostgreSQL conversation storage (LlamaIndex memory disabled)",
                       message_count=len(conversation_history))
        return

    async def _get_conversation_context(self, conversation_id: str) -> str:
        """Get conversation context for the given conversation ID using PostgreSQL."""
        if not self.conversation_service:
            return ""

        try:
            return await self.conversation_service.get_conversation_context(
                conversation_id,
                max_messages=10,  # Last 5 exchanges
                include_sources=False
            )
        except Exception as e:
            logger.error("Failed to get conversation context", error=str(e))
            return ""


    def _clean_conversation_context(self, raw_context: str) -> str:
        """Clean conversation context to remove meta-commentary and focus on substance."""
        if not raw_context:
            return ""

        lines = raw_context.split('\n')
        cleaned_lines = []

        # Patterns that indicate meta-commentary rather than actual answers
        meta_commentary_patterns = [
            r'^(Thoughtful|Excellent|Great|Fantastic|Well done|This is a|Perfect|Outstanding)',
            r'(comprehensive|well-structured|detailed|thorough) response',
            r'(analysis|breakdown|explanation) you\'ve provided',
            r'quality.*response',
            r'addresses.*question.*well',
            r'clear.*comprehensive.*answer',
            r'excellent.*coverage'
        ]

        import re

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Skip lines that are meta-commentary
            is_meta_commentary = False
            for pattern in meta_commentary_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    is_meta_commentary = True
                    break

            # Also skip very short responses that look like feedback
            if len(line) < 20 and any(word in line.lower() for word in ['great', 'excellent', 'good', 'nice', 'perfect']):
                is_meta_commentary = True

            if not is_meta_commentary:
                cleaned_lines.append(line)

        cleaned_context = '\n'.join(cleaned_lines)

        # Limit context length to prevent overwhelming the agents
        if len(cleaned_context) > 2000:
            # Take the most recent 2000 characters
            cleaned_context = cleaned_context[-2000:]
            # Try to start at a complete message boundary
            user_pos = cleaned_context.find('User:')
            if user_pos > 0:
                cleaned_context = cleaned_context[user_pos:]

        return cleaned_context

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
        """
        Process user query using intelligent agentic RAG system with CrewAI agents.
        
        Args:
            question (str): User's question or query
            conversation_history (List[Dict]): Previous conversation messages
            conversation_id (Optional[str]): Unique conversation identifier
            
        Returns:
            Dict[str, Any]: Response containing answer, sources, and metadata
            
        Raises:
            RuntimeError: If service not initialized
        """
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

            # Memory is now handled by PostgreSQL only
            # await self._populate_memory_from_history(conversation_history)  # Disabled

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

            # Use CrewAI agents
            if self.crew_agents:
                try:
                    logger.info("🚀 Starting CrewAI agent processing",
                               question=question[:50],
                               has_crew_agents=True)

                    # Always provide context if we have conversation history
                    # Let the agents decide what's relevant vs what's a new topic
                    if conversation_history and len(conversation_history) > 0:
                        # Only get the last exchange for context
                        last_exchange = []
                        for msg in reversed(conversation_history[-4:]):  # Last 2 exchanges max
                            last_exchange.insert(0, msg)
                            if len(last_exchange) >= 4:  # 2 full exchanges
                                break

                        context_str = ""
                        for msg in last_exchange:
                            role = msg.get("role", "")
                            content = msg.get("content", "")
                            if role and content:
                                context_str += f"{role.title()}: {content[:200]}...\n"

                        query_to_process = f"""CONVERSATION CONTEXT:
{context_str}

CURRENT QUESTION: {question}

Instructions: This question may be related to our previous conversation or it may be completely new. Analyze the context and current question to determine the best approach."""
                        logger.info("🔗 Adding conversation context to query")
                    else:
                        # No conversation history available
                        query_to_process = question
                        logger.info("🆕 Processing as new question (no conversation history)")

                    # Process with timeout protection
                    result = await asyncio.wait_for(
                        self.crew_agents.process_query(query_to_process),
                        timeout=120  # 2 minute hard timeout
                    )

                    # Add to conversation history if conversation_id provided
                    if conversation_id:
                        await self._add_to_conversation(
                            conversation_id, "user", question, processing_mode="crew_ai"
                        )
                        await self._add_to_conversation(
                            conversation_id, "assistant", result['response'],
                            sources=result.get('sources', []),
                            processing_mode="crew_ai_response"
                        )

                    # Ensure metadata is properly set
                    if "metadata" not in result:
                        result["metadata"] = {}
                    result["metadata"].update({
                        "model": self.config.ollama_model,
                        "conversation_id": conversation_id,
                        "source_count": len(result.get('sources', [])),
                        "processing_mode": "crew_ai",
                        "has_conversation_context": len(conversation_history) > 0
                    })

                    # Log to Phoenix if available
                    if self.phoenix_service:
                        await self.phoenix_service.log_chat_interaction(
                            conversation_id or "unknown",
                            question,
                            result["response"],
                            result.get('sources', []),
                            result.get("metadata", {})
                        )

                    return result

                except Exception as e:
                    logger.error("CrewAI agents failed", error=str(e))
                    # Return error response instead of falling back
                    result = {
                        "response": "I apologize, but I'm currently experiencing issues with the AI agents. Please try again in a moment.",
                        "sources": [],
                        "metadata": {
                            "model": self.config.ollama_model,
                            "conversation_id": conversation_id,
                            "source_count": 0,
                            "processing_mode": "crew_ai_error",
                            "error": str(e)
                        }
                    }
                    return result


            # Backup query engine option (if needed) - COMMENTED OUT
            # logger.info("Using backup query engine")
            # response = await asyncio.to_thread(
            #     self.query_engine.query,
            #     question
            # )

            # Extract source information - COMMENTED OUT
            # sources = []
            # if hasattr(response, 'source_nodes') and response.source_nodes:
            #     for node in response.source_nodes:
            #         source_info = {
            #             "content": node.text[:200] + "..." if len(node.text) > 200 else node.text,
            #             "score": float(node.score) if hasattr(node, 'score') else 1.0,
            #             "metadata": node.metadata
            #         }
            #         sources.append(source_info)

            # response_text = str(response.response)

            # result = {
            #     "response": response_text,
            #     "sources": sources,
            #     "metadata": {
            #         "model": self.config.ollama_model,
            #         "conversation_id": conversation_id,
            #         "source_count": len(sources),
            #         "processing_mode": "fallback_query_engine",
            #         "has_conversation_context": len(conversation_history) > 0
            #     }
            # }

            # This should never be reached since we only use CrewAI agents now
            logger.error("No agents available for processing")
            return {
                "response": "System configuration error - no processing method available",
                "sources": [],
                "metadata": {
                    "model": self.config.ollama_model,
                    "conversation_id": conversation_id,
                    "source_count": 0,
                    "processing_mode": "configuration_error"
                }
            }

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
                logger.warning("Conversation service not available")
                return []

        except Exception as e:
            logger.error("Failed to get conversation history", error=str(e))
            return []
    
    async def clear_conversation(self, conversation_id: str) -> bool:
        """Clear conversation history from PostgreSQL."""
        try:
            if self.conversation_service:
                result = await self.conversation_service.clear_conversation(conversation_id)
                logger.info("Conversation cleared", conversation_id=conversation_id, success=result)
                return result
            else:
                logger.warning("Conversation service not available")
                return False
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
    
    async def get_available_prompts(self) -> List[Dict[str, Any]]:
        """Get available prompts from Phoenix service."""
        try:
            if self.phoenix_service:
                prompts = await self.phoenix_service.get_available_prompts()
                return [
                    {
                        "id": prompt.id,
                        "name": prompt.name,
                        "description": prompt.description,
                        "variables": prompt.variables,
                        "tags": prompt.tags
                    }
                    for prompt in prompts
                ]
            return []
        except Exception as e:
            logger.error("Failed to get available prompts", error=str(e))
            return []
    
    async def render_prompt_with_phoenix(self, prompt_id: str, variables: Dict[str, str]) -> Optional[str]:
        """Render a prompt using Phoenix service."""
        try:
            if self.phoenix_service:
                return await self.phoenix_service.render_prompt(prompt_id, variables)
            return None
        except Exception as e:
            logger.error("Failed to render prompt", prompt_id=prompt_id, error=str(e))
            return None
    
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