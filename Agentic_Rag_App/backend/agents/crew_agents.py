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
from llama_index.embeddings.gemini import GeminiEmbedding
from config import BackendConfig

# Explicitly disable OpenAI for CrewAI to prevent API key errors
os.environ['OPENAI_API_KEY'] = ''
os.environ['OPENAI_API_BASE'] = ''

# Explicitly disable Vertex AI and force standard Gemini API
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ''
os.environ['VERTEX_AI_PROJECT'] = ''

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

        # Configure LLM for agents based on provider
        self.llm = self._configure_llm()

        self._initialize_agents()

    def _configure_llm(self):
        """Configure LLM based on provider (Ollama or Gemini)."""
        if self.config.llm_provider == 'gemini':
            logger.info("Configuring Gemini LLM for CrewAI agents")
            # Force standard Gemini API by setting provider environment
            import os
            os.environ['GEMINI_API_KEY'] = self.config.gemini_api_key

            return LLM(
                model=f"gemini/{self.config.gemini_model}",  # Use gemini/ prefix
                api_key=self.config.gemini_api_key,
                temperature=0.0,
                max_tokens=2000,
                timeout=30
            )
        else:  # Default to Ollama
            logger.info("Configuring Ollama LLM for CrewAI agents")
            return LLM(
                model=f"ollama/{self.config.ollama_model}",
                api_base=self.config.ollama_base_url,
                temperature=0.0,
                max_tokens=2000,
                timeout=30
            )

    def _create_document_retrieval_tool(self):
        """Create a document retrieval tool using the @tool decorator."""

        @tool("Search Documents")
        def search_documents(search_terms: str) -> str:
            """Search for relevant documents using provided search terms.

            Args:
                search_terms: Simple search keywords as a string
            """
            print(f"🔍 DEBUG LINE 98: search_documents function called with: {search_terms[:100]}")
            try:
                logger.info("🔍 CREW AI TOOL CALLED: search_documents", search_terms=search_terms[:100], provider=self.config.llm_provider)
                print(f"🔍 DEBUG LINE 99: Logger info executed")

                # Validate we have a proper query string
                if not search_terms or not search_terms.strip():
                    print("❌ DEBUG LINE 102: Empty search terms")
                    return "Error: Search query cannot be empty."

                search_query = search_terms.strip()
                print(f"🔍 DEBUG LINE 106: Search query processed: {search_query}")

                # Check if we got a placeholder description instead of real query
                placeholder_queries = [
                    "The search query to find relevant documents",
                    "Search query",
                    "query",
                    "search"
                ]
                if search_query.lower() in [p.lower() for p in placeholder_queries]:
                    print("❌ DEBUG LINE 115: Placeholder query detected")
                    return "Error: Please provide a specific search query."

                print(f"✅ DEBUG LINE 118: Query validation passed")
                
                # Use your RAG service's existing database configuration
                DATABASE_URL = self.config.database_url
                db_url_parts = urlparse(DATABASE_URL)
                print(f"🔗 DEBUG LINE 122: Database URL: {DATABASE_URL}")
                print(f"🔗 DEBUG LINE 123: DB Host: {db_url_parts.hostname}, Port: {db_url_parts.port}")

                logger.info("Using RAG service database connection",
                           host=db_url_parts.hostname,
                           port=db_url_parts.port,
                           database=db_url_parts.path.lstrip('/'),
                           user=db_url_parts.username)
                print(f"🔗 DEBUG LINE 130: Logger database info executed")
                
                # Initialize the vector store with your configuration
                print(f"🗄️ DEBUG LINE 131: Creating vector store with table: embeddings_gemini")
                vector_store = PGVectorStore.from_params(
                    host=db_url_parts.hostname,
                    port=db_url_parts.port,
                    database=db_url_parts.path.lstrip('/'),
                    user=db_url_parts.username,
                    password=db_url_parts.password,
                    table_name="embeddings_gemini",
                    embed_dim=768,  # Match the actual database embedding dimensions
                )
                print(f"✅ DEBUG LINE 140: Vector store created successfully")

                # Initialize embedding model based on provider
                print(f"🤖 DEBUG LINE 142: LLM Provider: {self.config.llm_provider}")
                if self.config.llm_provider == 'gemini':
                    print(f"🤖 DEBUG LINE 144: Creating Gemini embedding model: {self.config.gemini_embedding_model}")
                    embed_model = GeminiEmbedding(
                        model_name=self.config.gemini_embedding_model,
                        api_key=self.config.gemini_api_key,
                    )
                    logger.info("Using Gemini embedding model for document retrieval")
                    print(f"✅ DEBUG LINE 150: Gemini embedding model created")
                else:
                    print(f"🤖 DEBUG LINE 152: Creating Ollama embedding model: {self.config.ollama_embedding_model}")
                    embed_model = OllamaEmbedding(
                        model_name=self.config.ollama_embedding_model,
                        base_url=self.config.ollama_base_url,
                    )
                    logger.info("Using Ollama embedding model for document retrieval")
                    print(f"✅ DEBUG LINE 158: Ollama embedding model created")

                # Create a LlamaIndex VectorStoreIndex object from the vector store
                print(f"📚 DEBUG LINE 163: Creating VectorStoreIndex from vector store")
                index = VectorStoreIndex.from_vector_store(
                    vector_store=vector_store,
                    embed_model=embed_model
                )
                print(f"✅ DEBUG LINE 167: VectorStoreIndex created successfully")

                print(f"🔍 DEBUG LINE 169: Creating retriever with top_k={self.config.similarity_top_k}")
                retriever = index.as_retriever(
                    similarity_top_k=self.config.similarity_top_k,
                    verbose=False  # Turn off verbose to prevent tool output leakage
                )
                print(f"✅ DEBUG LINE 173: Retriever created successfully")

                # Query the index to retrieve nodes directly
                print(f"📊 DEBUG LINE 175: About to query vector database with: '{search_query}'")
                logger.info("📊 Querying vector database", query=search_query, table_suffix="gemini" if self.config.llm_provider == 'gemini' else "ollama")
                retrieved_nodes = retriever.retrieve(search_query)
                print(f"✅ DEBUG LINE 178: Retrieved {len(retrieved_nodes)} nodes from database")

                # Print detailed info about each retrieved node
                for i, node in enumerate(retrieved_nodes):
                    score = getattr(node, 'score', 0.0)
                    content_preview = node.text[:200] + "..." if len(node.text) > 200 else node.text
                    source_info = node.metadata.get('filename', 'Unknown') if hasattr(node, 'metadata') and node.metadata else 'No metadata'
                    print(f"📄 DEBUG NODE {i+1}: Score={score:.3f}, Source={source_info}")
                    print(f"📄 DEBUG CONTENT {i+1}: {content_preview}")
                    print("---")
                logger.info("✅ Database retrieval completed", num_results=len(retrieved_nodes), has_results=len(retrieved_nodes) > 0)

                if not retrieved_nodes:
                    print(f"❌ DEBUG LINE 181: No nodes retrieved, returning error message")
                    return f"No relevant documents found for query: '{search_query}'. Please try different keywords or check if documents are properly indexed."
                
                # Format the retrieved context with source metadata - Keep concise for Gemma2:1b
                formatted_chunks = []
                for i, node in enumerate(retrieved_nodes, 1):
                    content = node.text[:800]  # Limit content size
                    
                    # Extract source file information from metadata
                    source_info = "Unknown source"
                    page_info = ""
                    
                    if hasattr(node, 'metadata') and node.metadata:
                        file_name = node.metadata.get('filename', 'Unknown file')
                        source_info = f"Source: {file_name}"
                        
                        page_num = node.metadata.get('page_label', '')
                        if page_num:
                            page_info = f" (Page {page_num})"
                    
                    score = getattr(node, 'score', 0.0)
                    formatted_chunk = f"""DOCUMENT {i}:
{source_info}{page_info} | Score: {score:.2f}
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

        # Agent 1: Document Retrieval Agent
        self.retrieval_agent = Agent(
            role="Document Retrieval Specialist",
            goal="Find relevant documents from the knowledge base for each specific query",
            backstory="""You are a document search specialist. CRITICAL RULES:

            FOLLOW-UP QUESTIONS (like "list in X points", "summarize", "tell me more"):
            - DO NOT search documents again
            - Return message: "Follow-up question detected - using previous documents"

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
            max_iter=5,
            max_execution_time=60
        )

        # Agent 2: Response Generation Agent
        self.response_agent = Agent(
            role="Answer Generator",
            goal="Write one clear answer using the retrieved documents",
            backstory="You read documents and write one clear answer. Only use information from the documents provided. Write directly without mentioning agent names or processes.",
            llm=self.llm,
            verbose=False,
            allow_delegation=False,
            max_iter=5,
            max_execution_time=60
        )
        
    
    def create_crew(self, query: str) -> Crew:
        """Create a simple 2-agent crew for processing queries."""

        # Task 1: Find relevant documents
        retrieval_task = Task(
            description=f"""Find relevant documents for this query: {query}

            IMPORTANT - Domain Recognition:
            1. If query contains CONVERSATION CONTEXT, use the context to understand the topic
            2. For new questions, identify the domain:
               - HR/Personnel questions (penalties, disciplinary, violations, infringements, employee rights, disciplinary actions) → Use "HR bylaws disciplinary penalties violations" search terms
               - Procurement questions (tendering, suppliers, sourcing, procurement, RFP, RFQ) → Use "procurement tendering" search terms
               - Security questions → Use "security" search terms
            3. Use domain-appropriate keywords in your search

            Use the Search Documents tool with the right domain keywords to find relevant information.""",
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
            max_execution_time=180
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
            logger.info("Starting CrewAI processing",
                       query=query[:100],
                       has_context="CONVERSATION CONTEXT:" in query)

            # Create crew for query processing
            crew = self.create_crew(query)
            logger.info("CrewAI agents initialized", agent_count=len(crew.agents))

            # Execute the crew workflow
            logger.info("🚀 STARTING CrewAI multi-agent workflow", provider=self.config.llm_provider)
            result = crew.kickoff()
            logger.info("🎯 CrewAI workflow completed successfully")

            # Extract ONLY the final task's output, not the entire workflow
            if hasattr(result, 'tasks_output') and result.tasks_output:
                # Get the last task's output (response_task)
                raw_output = str(result.tasks_output[-1].raw)
            elif hasattr(result, 'raw'):
                raw_output = str(result.raw)
            else:
                raw_output = str(result)

            logger.info("Raw output extracted", length=len(raw_output), preview=raw_output[:100])

            # Clean the CrewAI response as fallback
            # crew_ai_response = self._clean_response(raw_output, query)  # Commented out to see raw chunks with scores
            crew_ai_response = raw_output

            # DISABLED: Context summary generation - using only tool calling now
            # intelligent_response, sources = self._generate_summary_from_context(query)

            # Force use of CrewAI response only (tool calling)
            final_response = crew_ai_response
            sources = []  # Let CrewAI agents handle sources through tool calling
            logger.info("Using pure CrewAI response from tool calling only")
            
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
                    "model": f"crewai-agentic-rag-{self.config.llm_provider}-{self.config.gemini_model if self.config.llm_provider == 'gemini' else self.config.ollama_model}",
                    "agents_used": ["intelligent_retrieval_specialist", "information_extractor"],  # Only 2 agents now
                    "process_type": "sequential",
                    "query_type": query_type,
                    "source_count": len(sources),
                    "llm_provider": self.config.llm_provider
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
                    "model": f"crewai-agentic-rag-{self.config.llm_provider}-{self.config.gemini_model if self.config.llm_provider == 'gemini' else self.config.ollama_model}",
                    "process_type": "error_handling",
                    "query_type": "error",
                    "llm_provider": self.config.llm_provider
                }
            }

    def _generate_summary_from_context(self, query: str):
        """Generate intelligent response with sources from knowledge base."""
        sources = []
        final_response = None

        try:
            # Get documents from knowledge base
            retriever = self.rag_service.index.as_retriever(
                similarity_top_k=self.config.similarity_top_k
            )
            nodes = retriever.retrieve(query)

            if nodes:
                # Store all sources with metadata
                for node in nodes:
                    source_info = {
                        "content": node.text[:800] + "..." if len(node.text) > 800 else node.text,
                        "score": float(getattr(node, 'score', 0.0)),
                        "metadata": node.metadata if hasattr(node, 'metadata') else {}
                    }
                    sources.append(source_info)

                # Generate summary from context
                combined_content = "\n\n".join([node.text[:800] for node in nodes[:3]])

                summary_prompt = f"""Answer the question using ONLY the provided documents. If the documents don't contain relevant information for the specific question asked, respond with "I don't have relevant information about this topic in the available documents."

Question: "{query}"

Retrieved Content:
{combined_content}

Instructions:
- Provide a detailed, well-structured answer using only relevant document content
- A REFORMAT REQUEST: User wants to modify/reformat a previous response (e.g., "give answer in 5 lines", "summarize above", "make it shorter")
   - If so: Use the conversation context to reformat the previous assistant response
   - Do NOT search for new documents
- Be comprehensive and thorough in your explanation
- Include relevant details, examples, and context from the documents
- Write as a natural, flowing response
- Don't mention document numbers or sources in the answer

Answer:"""

                # Use the same LLM to create a fine-tuned summary
                try:
                    # Try different methods to call the LLM
                    if hasattr(self.llm, 'call'):
                        summary_result = self.llm.call(summary_prompt)
                    elif hasattr(self.llm, '__call__'):
                        summary_result = self.llm(summary_prompt)
                    elif hasattr(self.llm, 'generate'):
                        summary_result = self.llm.generate(summary_prompt)
                    else:
                        # Fallback - try direct call
                        summary_result = self.llm(summary_prompt)

                    if hasattr(summary_result, 'content'):
                        response = summary_result.content.strip()
                    elif hasattr(summary_result, 'text'):
                        response = summary_result.text.strip()
                    else:
                        response = str(summary_result).strip()
                except Exception as llm_error:
                    logger.warning("LLM call failed, using simple concatenation", error=str(llm_error))
                    # Fallback to simple content concatenation
                    response = combined_content[:1000]

                # Clean the response
                final_response = self._clean_response(response, query)

                if final_response:
                    logger.info("Generated intelligent response", source_count=len(sources))

        except Exception as e:
            logger.warning("Could not generate response from context", error=str(e))

        return final_response, sources

# Simple test function
def test_basic_functionality():
    """Basic test for RAG Crew functionality."""
    print("Basic RAG Crew test - run this to verify setup")
    from config import BackendConfig
    config = BackendConfig()
    print(f"Config loaded: {config.ollama_base_url}")


if __name__ == "__main__":
    test_basic_functionality()