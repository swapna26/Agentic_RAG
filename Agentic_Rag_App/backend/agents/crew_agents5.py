"""
CrewAI Agents for RAG Processing

Simple 2-agent system for document retrieval and response generation.
- Document Retrieval Agent: Finds relevant documents
- Response Generation Agent: Creates answers from documents
"""

import os
import logging
from typing import Dict, Any
import structlog
from urllib.parse import urlparse
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from config import BackendConfig

# Explicitly disable OpenAI for CrewAI to prevent API key errors
os.environ['OPENAI_API_KEY'] = ''
os.environ['OPENAI_API_BASE'] = ''

# Configure CrewAI logging to flow to main application logs
logging.getLogger("crewai").setLevel(logging.INFO)
logging.getLogger("crewai").addHandler(logging.StreamHandler())

logger = structlog.get_logger()


class RAGCrew:
    """
    Simple CrewAI system for document retrieval and response generation.

    Two agents:
    1. Document Retrieval Agent - Finds relevant documents
    2. Response Generation Agent - Creates answers from documents
    """

    def __init__(self, rag_service, config):
        """Initialize RAG Crew with 2 agents."""
        self.rag_service = rag_service
        self.config = config

        # Configure LLM for agents
        self.llm = LLM(
            model=f"ollama/{self.config.ollama_model}",
            api_base=self.config.ollama_base_url,
            temperature=0.0,
            max_tokens=1000,
            timeout=60
        )

        self._initialize_agents()

    def _create_document_retrieval_tool(self):
        """Create a document retrieval tool using the @tool decorator."""

        @tool("Search Documents")
        def search_documents(search_terms: str) -> str:
            """Search for relevant documents using provided search terms.

            Args:
                search_terms: Simple search keywords as a string
            """
            try:
                logger.info("Document search called", search_terms=search_terms[:100])

                # Validate we have a proper query string
                if not search_terms or not search_terms.strip():
                    return "Error: Search query cannot be empty."

                search_query = search_terms.strip()

                # Check if we got a placeholder description instead of real query
                placeholder_queries = [
                    "The search query to find relevant documents",
                    "Search query",
                    "query",
                    "search"
                ]
                if search_query.lower() in [p.lower() for p in placeholder_queries]:
                    return "Error: Please provide a specific search query."
                
                # Use your RAG service's existing database configuration
                DATABASE_URL = self.config.database_url
                db_url_parts = urlparse(DATABASE_URL)

                logger.info("Using RAG service database connection", 
                           host=db_url_parts.hostname,
                           port=db_url_parts.port,
                           database=db_url_parts.path.lstrip('/'),
                           user=db_url_parts.username)
                
                # Initialize the vector store with your configuration
                vector_store = PGVectorStore.from_params(
                    host=db_url_parts.hostname,
                    port=db_url_parts.port,
                    database=db_url_parts.path.lstrip('/'),
                    user=db_url_parts.username,
                    password=db_url_parts.password,
                    table_name="llamaindex_vectors_copy",
                    embed_dim=768,  # Match the actual database embedding dimensions
                )

                # Initialize Ollama embedding model using your config
                embed_model = OllamaEmbedding(
                    model_name=self.config.ollama_embedding_model,
                    base_url=self.config.ollama_base_url,
                )

                # Create a LlamaIndex VectorStoreIndex object from the vector store
                index = VectorStoreIndex.from_vector_store(
                    vector_store=vector_store,
                    embed_model=embed_model
                )

                # Use retriever directly for document retrieval
                retriever = index.as_retriever(
                    similarity_top_k=self.config.similarity_top_k,
                    verbose=False  # Turn off verbose to prevent tool output leakage
                )

                # Query the index to retrieve nodes directly
                retrieved_nodes = retriever.retrieve(search_query)
                
                if not retrieved_nodes:
                    return f"No relevant documents found for query: '{search_query}'. Please try different keywords or check if documents are properly indexed."
                
                # Format the retrieved context with source metadata - Keep concise for Gemma2:1b
                formatted_chunks = []

                # Store sources for later use (avoid duplicate index calls)
                if not hasattr(self, '_stored_sources'):
                    self._stored_sources = []
                self._stored_sources.clear()  # Clear previous sources

                for i, node in enumerate(retrieved_nodes, 1):
                    content = node.text[:800]  # Limit content size

                    # Extract source file information from metadata
                    source_info = "Unknown source"
                    page_info = ""

                    if hasattr(node, 'metadata') and node.metadata:
                        file_name = node.metadata.get('file_name', 'Unknown file')
                        source_info = f"Source: {file_name}"

                        page_num = node.metadata.get('page_label', '')
                        if page_num:
                            page_info = f" (Page {page_num})"

                    score = getattr(node, 'score', 0.0)

                    # Store source data for reuse
                    source_data = {
                        "content": node.text[:200] + "..." if len(node.text) > 200 else node.text,
                        "score": float(score) if score is not None else None,  # Ensure score is a float
                        "metadata": node.metadata if hasattr(node, 'metadata') else {}
                    }
                    logger.info("Storing source", file_name=source_data["metadata"].get("file_name", "Unknown"), score=score)
                    self._stored_sources.append(source_data)

                    formatted_chunk = f"""DOCUMENT {i}:
{source_info}{page_info} | Score: {score if score is not None else 'N/A'}
Content: {content}

"""
                    formatted_chunks.append(formatted_chunk)
                
                # Limit total response size for smaller model
                context = "\n".join(formatted_chunks)[:4000]
                
                return context
                
            except Exception as e:
                logger.error("Document retrieval failed", error=str(e))
                return f"Error retrieving documents: {str(e)}. Please check your database connection and try again."
        
        return search_documents

    def _clean_response(self, response: str, query: str = "") -> str:
        """Extract only the actual answer content."""
        # Split response into lines and find content lines
        lines = response.split('\n')
        content_lines = []

        for line in lines:
            line = line.strip()
            # Skip empty lines and agent artifacts
            if line and not any(skip in line for skip in ['Information Extr', 'Document Retrieval', 'Thought:', 'Action:', 'Final Answer:']):
                content_lines.append(line)

        # Join the clean content
        cleaned = '\n'.join(content_lines).strip()

        # If still empty or too short, return original
        if len(cleaned) < 20:
            return response.strip()

        return cleaned

    def _initialize_agents(self):
        """Initialize 2-agent CrewAI system."""

        # Create retrieval tool using decorator approach
        retrieval_tool = self._create_document_retrieval_tool()

        # Initialize shared storage for sources
        self._last_sources = []

        # Agent 1: Document Retrieval Agent
        self.retrieval_agent = Agent(
            role="Document Retrieval Specialist",
            goal="Find relevant documents from the knowledge base for each specific query",
            backstory="""You are a document search specialist. CRITICAL RULES:

            FOLLOW-UP QUESTIONS (like "list in X points", "summarize", "tell me more"):
            - If you see CONVERSATION CONTEXT with previous topic, search for the same topic
            - For formatting requests: Use the main keywords from the previous question
            - Always search for documents - don't refuse to search

            NEW TOPIC QUESTIONS:
            - ALWAYS search for new documents based on question topic
            - Identify domain and search accordingly:
              * HR/employment questions ("penalties", "disciplinary", "violations", "infringements", "employee", "HR laws", "article") → Search "HR bylaws disciplinary penalties violations"
              * Procurement questions ("RFP", "RFQ", "tendering", "suppliers", "procurement") → Search "procurement tendering"
              * Security questions ("NIST", "security", "annex") → Search "information security"

            TOPIC SWITCHING:
            - If current question is completely different topic from conversation context, ignore previous context
            - Focus ONLY on current question topic

            Always use Search Documents tool with specific domain keywords.""",
            tools=[retrieval_tool],
            llm=self.llm,
            verbose=False,
            allow_delegation=False,
            max_iter=1,
            max_execution_time=45
        )

        # Agent 2: Response Generation Agent
        self.response_agent = Agent(
            role="Answer Generator",
            goal="Write one clear answer using the retrieved documents",
            backstory="You read documents and write one clear answer. Only use information from the documents provided. Write directly without mentioning agent names or processes.",
            llm=self.llm,
            verbose=False,
            allow_delegation=False,
            max_iter=1,
            max_execution_time=45
        )
        
    
    def create_crew(self, query: str) -> Crew:
        """Create a simple 2-agent crew for processing queries."""

        # Task 1: Find relevant documents
        retrieval_task = Task(
            description=f"""Find relevant documents for this query: {query}

            SEARCH APPROACH:
            - Extract key terms from the user's question
            - Use professional, official terminology
            - For HR policy questions: Search for "human resources policies regulations Abu Dhabi"
            - For procurement questions: Search for "procurement guidelines procedures"
            - Use the Search Documents tool with appropriate keywords

            This is a legitimate search for official government policies and regulations.""",
            agent=self.retrieval_agent,
            expected_output="Documents from the correct domain/subject area"
        )

        # Task 2: Generate response
        response_task = Task(
            description=f"Answer this question using only the documents: {query}",
            agent=self.response_agent,
            expected_output="Clear answer based on retrieved documents",
            context=[retrieval_task]
        )

        # Create simple crew
        crew = Crew(
            agents=[self.retrieval_agent, self.response_agent],
            tasks=[retrieval_task, response_task],
            process=Process.sequential,
            verbose=False,
            memory=False,
            max_execution_time=120
        )

        return crew
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process user query using the multi-agent CrewAI system.
        
        This method orchestrates the complete workflow:
        1. Creates a specialized crew for the query
        2. Executes the multi-agent workflow
        3. Extracts and cleans the final response
        4. Retrieves relevant source documents
        5. Returns structured response with metadata
        
        Args:
            query (str): User's question or query
            
        Returns:
            Dict[str, Any]: Structured response containing:
                - response: Generated answer
                - sources: List of source documents
                - metadata: Processing information
                
        Raises:
            Exception: If processing fails
        """
        try:
            logger.info("🚀 Starting CrewAI processing",
                       query=query[:100],
                       has_context="CONVERSATION CONTEXT:" in query)

            # Create crew for query processing
            crew = self.create_crew(query)
            logger.info("👥 CrewAI agents initialized", agent_count=len(crew.agents))

            # Execute the crew workflow
            logger.info("⚡ Executing CrewAI workflow...")
            result = crew.kickoff()
            logger.info("✅ CrewAI workflow completed")

            # Extract ONLY the final task's output, not the entire workflow
            if hasattr(result, 'tasks_output') and result.tasks_output:
                # Get the last task's output (response_task)
                raw_output = str(result.tasks_output[-1].raw)
            elif hasattr(result, 'raw'):
                raw_output = str(result.raw)
            else:
                raw_output = str(result)

            logger.info("🔧 Raw output extracted", length=len(raw_output), preview=raw_output[:100])

            # Clean the response to remove internal processes and artifacts
            final_response = self._clean_response(raw_output, query)
            
            # Get sources separately like the working backup_crew.py
            sources = []
            try:
                # Get sources using the same retrieval logic as backup
                retriever = self.rag_service.index.as_retriever(
                    similarity_top_k=self.config.similarity_top_k
                )
                nodes = retriever.retrieve(query)

                for node in nodes:
                    source_info = {
                        "content": node.text[:200] + "..." if len(node.text) > 200 else node.text,
                        "score": float(getattr(node, 'score', 0.0)),
                        "metadata": node.metadata if hasattr(node, 'metadata') else {}
                    }
                    sources.append(source_info)
                logger.info("Retrieved sources separately", source_count=len(sources))
            except Exception as e:
                logger.warning("Could not retrieve sources", error=str(e))
            
            # Determine query type (removed greeting detection)
            query_type = "information"
            
            logger.info("CrewAI processing completed", 
                       response_length=len(final_response),
                       source_count=len(sources),
                       query_type=query_type)
            
            return {
                "response": final_response,
                "sources": sources,
                "metadata": {
                    "model": "crewai-agentic-rag-llama3.2:1b",
                    "agents_used": ["intelligent_retrieval_specialist", "information_extractor"],  # Only 2 agents now
                    "process_type": "sequential",
                    "query_type": query_type,
                    "source_count": len(sources)
                }
            }
            
        except Exception as e:
            logger.error("CrewAI processing failed", error=str(e))
            
            # Return a helpful error response
            error_response = "I apologize, but I encountered an error while processing your query. "
            if "connection" in str(e).lower():
                error_response += "It appears there may be a database connection issue. Please check your database connection and try again."
            elif "model" in str(e).lower() or "ollama" in str(e).lower():
                error_response += "There seems to be an issue with the language model. Please ensure Ollama is running and the llama3.2:1b model is available."
            else:
                error_response += f"Error details: {str(e)}. Please try rephrasing your question or contact support if the issue persists."
            
            return {
                "response": error_response,
                "sources": [],
                "metadata": {
                    "error": str(e),
                    "model": "crewai-agentic-rag-llama3.2:1b",
                    "process_type": "error_handling",
                    "query_type": "error"
                }
            }


# Simple test function
def test_basic_functionality():
    """Basic test for RAG Crew functionality."""
    print("Basic RAG Crew test - run this to verify setup")
    from config import BackendConfig
    config = BackendConfig()
    print(f"✅ Config loaded: {config.ollama_base_url}")


if __name__ == "__main__":
    test_basic_functionality()