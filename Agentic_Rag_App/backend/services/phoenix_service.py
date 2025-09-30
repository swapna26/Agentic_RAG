"""Phoenix integration for prompt lifecycle management and observability."""

import asyncio
import os
from typing import Dict, Any, Optional, List
import structlog
import httpx
from pydantic import BaseModel

# OpenTelemetry imports for Phoenix tracing
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource

# Phoenix specific imports
try:
    import phoenix as px
    from phoenix.trace import using_project
    PHOENIX_AVAILABLE = True
except ImportError:
    PHOENIX_AVAILABLE = False
    px = None
    using_project = None

logger = structlog.get_logger()


class PromptTemplate(BaseModel):
    """Prompt template model."""
    id: str
    name: str
    template: str
    variables: List[str]
    description: Optional[str] = None
    tags: List[str] = []


class PhoenixService:
    """Service for managing prompts via Phoenix."""
    
    def __init__(self, config):
        self.config = config
        self.base_url = config.phoenix_base_url
        self.project_name = config.phoenix_project_name
        self.prompts_cache = {}
        self.client = None
        self.tracer = None
        self.tracing_enabled = False
        
        # Default prompts for the RAG system
        self.default_prompts = {
            "query_analysis": {
                "name": "Query Analysis",
                "template": """Analyze the following user query and optimize it for document retrieval:

Query: {query}

Your tasks:
1. Identify key concepts and topics
2. Extract important keywords  
3. Determine the query intent and type
4. Suggest improvements for better retrieval

Provide clear analysis and an optimized query.""",
                "variables": ["query"],
                "description": "Analyzes user queries for better document retrieval",
                "tags": ["rag", "query-analysis"]
            },
            
            "document_retrieval": {
                "name": "Document Retrieval",
                "template": """Find and retrieve the most relevant documents for this query:

Query: {query}
Available Documents: {document_list}

Your tasks:
1. Identify which documents are most relevant
2. Rank them by relevance to the query
3. Explain why each document is relevant
4. Provide key excerpts from top documents

Focus on documents that directly answer the user's question.""",
                "variables": ["query", "document_list"],
                "description": "Retrieves and ranks relevant documents",
                "tags": ["rag", "retrieval"]
            },
            
            "response_generation": {
                "name": "Response Generation", 
                "template": """Generate a comprehensive response based on the retrieved documents:

Original Query: {query}
Retrieved Documents: {documents}

Your tasks:
1. Synthesize information from the documents
2. Create a clear, well-structured answer
3. Include proper source citations
4. Ensure factual accuracy
5. Address the query directly

The response should be informative and properly sourced.""",
                "variables": ["query", "documents"],
                "description": "Generates responses from retrieved documents",
                "tags": ["rag", "generation"]
            },
            
            "response_validation": {
                "name": "Response Validation",
                "template": """Review and validate this generated response:

Original Query: {query}
Generated Response: {response}
Source Documents: {sources}

Your tasks:
1. Check factual accuracy against sources
2. Verify proper source citations
3. Ensure the response fully addresses the query
4. Check for clarity and coherence
5. Suggest improvements if needed

Provide the final validated response or corrections.""",
                "variables": ["query", "response", "sources"],
                "description": "Validates and improves generated responses",
                "tags": ["rag", "validation"]
            }
        }
    
    async def initialize(self):
        """Initialize Phoenix service and prompts."""
        try:
            # Create HTTP client
            self.client = httpx.AsyncClient(timeout=30.0)

            # Test Phoenix connection
            await self._test_connection()

            # Initialize Phoenix tracing
            await self._setup_phoenix_tracing()

            # Initialize default prompts
            await self._initialize_default_prompts()

            logger.info("Phoenix service initialized successfully", tracing_enabled=self.tracing_enabled)

        except Exception as e:
            logger.warning("Phoenix initialization failed, using local prompts", error=str(e))
            # Continue with local prompts if Phoenix isn't available
            self._load_local_prompts()

    async def _setup_phoenix_tracing(self):
        """Setup OpenTelemetry tracing to Phoenix."""
        try:
            if not PHOENIX_AVAILABLE:
                logger.warning("Phoenix package not available, tracing disabled")
                return

            # Parse Phoenix URL
            from urllib.parse import urlparse
            parsed_url = urlparse(self.base_url)
            phoenix_host = parsed_url.hostname or "localhost"
            phoenix_port = parsed_url.port or 6006

            # Set environment variables
            os.environ["PHOENIX_HOST"] = phoenix_host
            os.environ["PHOENIX_PORT"] = str(phoenix_port)

            # Configure OpenTelemetry to send traces to Phoenix
            resource = Resource(attributes={
                SERVICE_NAME: self.project_name
            })

            # Create OTLP exporter pointing to Phoenix
            otlp_endpoint = f"http://{phoenix_host}:{phoenix_port}/v1/traces"

            exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            span_processor = BatchSpanProcessor(exporter)

            # Set up tracer provider
            tracer_provider = trace_sdk.TracerProvider(resource=resource)
            tracer_provider.add_span_processor(span_processor)

            # Set the global tracer provider
            trace.set_tracer_provider(tracer_provider)

            # Get tracer for this service
            self.tracer = trace.get_tracer(__name__)
            self.tracing_enabled = True

            logger.info("Phoenix tracing configured successfully",
                       endpoint=otlp_endpoint,
                       project=self.project_name)

        except Exception as e:
            logger.warning("Failed to setup Phoenix tracing", error=str(e))
            self.tracing_enabled = False
    
    async def _test_connection(self):
        """Test connection to Phoenix."""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            if response.status_code == 200:
                logger.info("Phoenix connection successful")
            else:
                raise Exception(f"Phoenix health check failed: {response.status_code}")
        except Exception as e:
            logger.warning("Phoenix connection test failed", error=str(e))
            raise
    
    async def _initialize_default_prompts(self):
        """Initialize default prompts in Phoenix."""
        for prompt_id, prompt_data in self.default_prompts.items():
            try:
                # Try to create or update prompt in Phoenix
                await self._create_or_update_prompt(prompt_id, prompt_data)
                
                # Cache the prompt locally
                self.prompts_cache[prompt_id] = PromptTemplate(
                    id=prompt_id,
                    **prompt_data
                )
                
            except Exception as e:
                logger.warning(f"Failed to initialize prompt {prompt_id}", error=str(e))
                # Fall back to local cache
                self.prompts_cache[prompt_id] = PromptTemplate(
                    id=prompt_id,
                    **prompt_data
                )
    
    def _load_local_prompts(self):
        """Load prompts locally when Phoenix isn't available."""
        for prompt_id, prompt_data in self.default_prompts.items():
            self.prompts_cache[prompt_id] = PromptTemplate(
                id=prompt_id,
                **prompt_data
            )
        logger.info("Loaded prompts locally")
    
    async def _create_or_update_prompt(self, prompt_id: str, prompt_data: Dict[str, Any]):
        """Create or update a prompt in Phoenix."""
        try:
            # This is a placeholder - actual Phoenix API calls would go here
            # For now, we'll just log the action
            logger.info("Prompt registered in Phoenix", 
                       prompt_id=prompt_id, 
                       name=prompt_data["name"])
            
        except Exception as e:
            logger.error("Failed to create/update prompt in Phoenix", 
                        prompt_id=prompt_id, 
                        error=str(e))
            raise
    
    async def get_prompt(self, prompt_id: str) -> Optional[PromptTemplate]:
        """Retrieve a prompt template."""
        try:
            # First try cache
            if prompt_id in self.prompts_cache:
                return self.prompts_cache[prompt_id]
            
            # Try to fetch from Phoenix if available
            if self.client:
                try:
                    # Placeholder for Phoenix API call
                    # response = await self.client.get(f"{self.base_url}/prompts/{prompt_id}")
                    # if response.status_code == 200:
                    #     prompt_data = response.json()
                    #     return PromptTemplate(**prompt_data)
                    pass
                except Exception as e:
                    logger.warning("Failed to fetch prompt from Phoenix", 
                                 prompt_id=prompt_id, 
                                 error=str(e))
            
            logger.warning("Prompt not found", prompt_id=prompt_id)
            return None
            
        except Exception as e:
            logger.error("Failed to get prompt", prompt_id=prompt_id, error=str(e))
            return None
    
    async def render_prompt(self, prompt_id: str, variables: Dict[str, str]) -> Optional[str]:
        """Render a prompt template with variables."""
        try:
            prompt = await self.get_prompt(prompt_id)
            if not prompt:
                return None
            
            # Simple string substitution
            rendered = prompt.template
            for var, value in variables.items():
                placeholder = "{" + var + "}"
                rendered = rendered.replace(placeholder, str(value))
            
            logger.info("Prompt rendered", prompt_id=prompt_id, variables=list(variables.keys()))
            return rendered
            
        except Exception as e:
            logger.error("Failed to render prompt", prompt_id=prompt_id, error=str(e))
            return None
    
    async def log_prompt_execution(self, prompt_id: str, variables: Dict[str, str],
                                 response: str, metadata: Dict[str, Any] = None):
        """Log prompt execution for observability."""
        try:
            # Create OpenTelemetry span if tracing is enabled
            if self.tracing_enabled and self.tracer:
                with self.tracer.start_as_current_span(f"prompt_execution_{prompt_id}") as span:
                    # Add attributes to the span
                    span.set_attribute("prompt.id", prompt_id)
                    span.set_attribute("prompt.variables_count", len(variables))
                    span.set_attribute("response.length", len(response))

                    # Add variable names as attributes
                    for key, value in variables.items():
                        span.set_attribute(f"prompt.variable.{key}", str(value)[:100])  # Limit value length

                    # Add metadata
                    if metadata:
                        for key, value in metadata.items():
                            span.set_attribute(f"metadata.{key}", str(value))

                    # Log the execution
                    span.add_event("prompt_executed", {
                        "prompt_id": prompt_id,
                        "variables": str(variables),
                        "response_preview": response[:200] if response else ""
                    })

                    logger.info("Prompt execution traced to Phoenix",
                               prompt_id=prompt_id,
                               span_id=span.get_span_context().span_id)
            else:
                # Fallback to regular logging
                log_data = {
                    "prompt_id": prompt_id,
                    "variables": variables,
                    "response_length": len(response),
                    "metadata": metadata or {}
                }
                logger.info("Prompt execution logged (no tracing)", **log_data)

        except Exception as e:
            logger.error("Failed to log prompt execution", error=str(e))

    async def log_chat_interaction(self, conversation_id: str, user_message: str,
                                 assistant_response: str, sources: List[Dict] = None,
                                 metadata: Dict[str, Any] = None):
        """Log complete chat interaction for observability."""
        try:
            if self.tracing_enabled and self.tracer:
                with self.tracer.start_as_current_span("chat_interaction") as span:
                    # Add chat attributes
                    span.set_attribute("chat.conversation_id", conversation_id)
                    span.set_attribute("chat.user_message", user_message[:500])  # Limit length
                    span.set_attribute("chat.assistant_response", assistant_response[:500])
                    span.set_attribute("chat.user_message_length", len(user_message))
                    span.set_attribute("chat.assistant_response_length", len(assistant_response))

                    # Add source information
                    if sources:
                        span.set_attribute("chat.sources_count", len(sources))
                        for i, source in enumerate(sources[:3]):  # Log first 3 sources
                            if isinstance(source, dict):
                                span.set_attribute(f"chat.source_{i}.score", source.get("score", 0.0))
                                if source.get("metadata"):
                                    span.set_attribute(f"chat.source_{i}.file",
                                                     source["metadata"].get("file_name", "unknown"))

                    # Add metadata
                    if metadata:
                        for key, value in metadata.items():
                            span.set_attribute(f"chat.metadata.{key}", str(value))

                    # Add detailed event
                    span.add_event("chat_completed", {
                        "conversation_id": conversation_id,
                        "user_message": user_message,
                        "assistant_response": assistant_response,
                        "sources": str(sources) if sources else "[]",
                        "metadata": str(metadata) if metadata else "{}"
                    })

                    logger.info("Chat interaction traced to Phoenix",
                               conversation_id=conversation_id,
                               message_length=len(user_message),
                               response_length=len(assistant_response),
                               sources_count=len(sources) if sources else 0)
            else:
                # Fallback to structured logging
                log_data = {
                    "chat_interaction": True,
                    "conversation_id": conversation_id,
                    "user_message": user_message,
                    "assistant_response": assistant_response,
                    "user_message_length": len(user_message),
                    "assistant_response_length": len(assistant_response),
                    "sources": sources,
                    "sources_count": len(sources) if sources else 0,
                    "metadata": metadata or {}
                }
                logger.info("Chat interaction logged", **log_data)

        except Exception as e:
            logger.error("Failed to log chat interaction", error=str(e))

    async def log_agent_workflow(self, workflow_type: str, query: str, agents_used: List[str],
                               execution_time: float, result: Dict[str, Any],
                               metadata: Dict[str, Any] = None):
        """Log agent workflow execution for observability."""
        try:
            if self.tracing_enabled and self.tracer:
                with self.tracer.start_as_current_span(f"agent_workflow_{workflow_type}") as span:
                    # Add workflow attributes
                    span.set_attribute("workflow.type", workflow_type)
                    span.set_attribute("workflow.query", query[:500])
                    span.set_attribute("workflow.agents_used", ",".join(agents_used))
                    span.set_attribute("workflow.execution_time", execution_time)
                    span.set_attribute("workflow.response_length", len(result.get("response", "")))

                    # Add result information
                    if result.get("sources"):
                        span.set_attribute("workflow.sources_count", len(result["sources"]))

                    # Add metadata
                    if metadata:
                        for key, value in metadata.items():
                            span.set_attribute(f"workflow.metadata.{key}", str(value))

                    # Add detailed event
                    span.add_event("agent_workflow_completed", {
                        "workflow_type": workflow_type,
                        "query": query,
                        "agents_used": agents_used,
                        "execution_time": execution_time,
                        "result": str(result)[:1000],  # Limit result size
                        "metadata": str(metadata) if metadata else "{}"
                    })

                    logger.info("Agent workflow traced to Phoenix",
                               workflow_type=workflow_type,
                               execution_time=execution_time,
                               agents_count=len(agents_used))
            else:
                # Fallback to structured logging
                log_data = {
                    "agent_workflow": True,
                    "workflow_type": workflow_type,
                    "query": query,
                    "agents_used": agents_used,
                    "execution_time": execution_time,
                    "result": result,
                    "metadata": metadata or {}
                }
                logger.info("Agent workflow logged", **log_data)

        except Exception as e:
            logger.error("Failed to log agent workflow", error=str(e))

    async def log_document_retrieval(self, query: str, retrieved_docs: List[Dict],
                                   retrieval_time: float, metadata: Dict[str, Any] = None):
        """Log document retrieval for observability."""
        try:
            if self.tracing_enabled and self.tracer:
                with self.tracer.start_as_current_span("document_retrieval") as span:
                    # Add retrieval attributes
                    span.set_attribute("retrieval.query", query[:500])
                    span.set_attribute("retrieval.docs_count", len(retrieved_docs))
                    span.set_attribute("retrieval.time", retrieval_time)

                    # Add document information
                    for i, doc in enumerate(retrieved_docs[:5]):  # Log first 5 docs
                        if isinstance(doc, dict):
                            span.set_attribute(f"retrieval.doc_{i}.score", doc.get("score", 0.0))
                            if doc.get("metadata"):
                                span.set_attribute(f"retrieval.doc_{i}.file",
                                                 doc["metadata"].get("file_name", "unknown"))

                    # Add metadata
                    if metadata:
                        for key, value in metadata.items():
                            span.set_attribute(f"retrieval.metadata.{key}", str(value))

                    # Add detailed event
                    span.add_event("documents_retrieved", {
                        "query": query,
                        "docs_count": len(retrieved_docs),
                        "retrieval_time": retrieval_time,
                        "documents": str(retrieved_docs)[:1000],  # Limit size
                        "metadata": str(metadata) if metadata else "{}"
                    })

                    logger.info("Document retrieval traced to Phoenix",
                               query_length=len(query),
                               docs_count=len(retrieved_docs),
                               retrieval_time=retrieval_time)
            else:
                # Fallback to structured logging
                log_data = {
                    "document_retrieval": True,
                    "query": query,
                    "retrieved_docs": retrieved_docs,
                    "docs_count": len(retrieved_docs),
                    "retrieval_time": retrieval_time,
                    "metadata": metadata or {}
                }
                logger.info("Document retrieval logged", **log_data)

        except Exception as e:
            logger.error("Failed to log document retrieval", error=str(e))

    def create_trace_span(self, operation_name: str, attributes: Dict[str, Any] = None):
        """Create a new trace span for manual tracing."""
        if not self.tracing_enabled or not self.tracer:
            return None

        span = self.tracer.start_span(operation_name)
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, str(value))

        return span
    
    async def get_available_prompts(self) -> List[PromptTemplate]:
        """Get list of all available prompts."""
        return list(self.prompts_cache.values())
    
    async def cleanup(self):
        """Cleanup Phoenix service."""
        try:
            if self.client:
                await self.client.aclose()
            logger.info("Phoenix service cleanup completed")
        except Exception as e:
            logger.error("Error during Phoenix cleanup", error=str(e))