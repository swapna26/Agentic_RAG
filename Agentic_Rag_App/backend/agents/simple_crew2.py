"""
Simple CrewAI Agents - Clean and Autonomous
No complex rules, minimal processing, agents understand their roles naturally
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

# Disable OpenAI for CrewAI
os.environ['OPENAI_API_KEY'] = ''
os.environ['OPENAI_API_BASE'] = ''

logger = structlog.get_logger()


class SimpleCrew:
    """Simple, clean CrewAI system with autonomous agents"""

    def __init__(self, rag_service, config):
        self.rag_service = rag_service
        self.config = config

        # Simple LLM configuration
        self.llm = LLM(
            model=f"ollama/{self.config.ollama_model}",
            api_base=self.config.ollama_base_url,
            temperature=0.1,
            max_tokens=800,
            timeout=30
        )

        self._setup_agents()

    def _create_document_search_tool(self):
        """Simple document search tool"""

        @tool("Search Documents")
        def search_documents(query: str) -> str:
            """Search for relevant documents using the query."""
            try:
                # Use the existing database configuration
                DATABASE_URL = self.config.database_url
                db_url_parts = urlparse(DATABASE_URL)

                # Initialize vector store
                vector_store = PGVectorStore.from_params(
                    host=db_url_parts.hostname,
                    port=db_url_parts.port,
                    database=db_url_parts.path.lstrip('/'),
                    user=db_url_parts.username,
                    password=db_url_parts.password,
                    table_name="llamaindex_vectors_copy",
                    embed_dim=768,
                )

                # Initialize embedding model
                embed_model = OllamaEmbedding(
                    model_name=self.config.ollama_embedding_model,
                    base_url=self.config.ollama_base_url,
                )

                # Create index and retrieve documents
                index = VectorStoreIndex.from_vector_store(
                    vector_store=vector_store,
                    embed_model=embed_model
                )

                retriever = index.as_retriever(similarity_top_k=5)
                retrieved_nodes = retriever.retrieve(query)

                if not retrieved_nodes:
                    return "No relevant documents found."

                # Format the results simply
                results = []
                for i, node in enumerate(retrieved_nodes, 1):
                    content = node.text[:600]  # Reasonable chunk size

                    # Get source info
                    source = "Unknown source"
                    if hasattr(node, 'metadata') and node.metadata:
                        file_name = node.metadata.get('file_name', 'Unknown file')
                        source = file_name

                        page_num = node.metadata.get('page_label', '')
                        if page_num:
                            source += f" (Page {page_num})"

                    score = getattr(node, 'score', 0.0)

                    # Store for later use
                    result_data = {
                        "content": content,
                        "source": source,
                        "score": score,
                        "metadata": node.metadata if hasattr(node, 'metadata') else {}
                    }

                    # Store in a way we can access later
                    if not hasattr(self, '_last_sources'):
                        self._last_sources = []
                    self._last_sources.append(result_data)

                    results.append(f"Document {i} - {source} (Score: {score:.2f}):\n{content}\n")

                return "\n".join(results)

            except Exception as e:
                logger.error("Document search failed", error=str(e))
                return f"Error searching documents: {str(e)}"

        return search_documents

    def _setup_agents(self):
        """Set up simple, autonomous agents"""

        # Create the document search tool
        search_tool = self._create_document_search_tool()

        # Agent 1: Document Finder - Simple and focused
        self.document_finder = Agent(
            role="Document Finder",
            goal="Find the most relevant documents for any question",
            backstory="""You are a document search specialist. Your job is simple:
            - Take any question and search for relevant documents
            - Use the search tool with appropriate keywords from the question
            - Find the best matching content""",
            tools=[search_tool],
            llm=self.llm,
            verbose=False,
            allow_delegation=False,
            max_iter=1,
            max_execution_time=30
        )

        # Agent 2: Answer Writer - Simple and direct
        self.answer_writer = Agent(
            role="Answer Writer",
            goal="Write clear answers using only the documents provided",
            backstory="""You are an answer writer. Your job is simple:
            - Read the documents provided by the document finder
            - Write a clear, helpful answer based only on those documents
            - If documents are relevant (good scores), extract and explain the information
            - If users ask for lists or points, format accordingly
            - Keep responses focused and factual""",
            llm=self.llm,
            verbose=False,
            allow_delegation=False,
            max_iter=1,
            max_execution_time=30
        )

    def _clean_response(self, response: str) -> str:
        """Minimal cleaning - remove only obvious technical artifacts"""
        import re

        # Remove only the most obvious technical leakage
        patterns_to_remove = [
            r"Action:.*?(?=\n|$)",
            r"Thought:.*?(?=\n|$)",
            r"Observation:.*?(?=\n|$)",
        ]

        cleaned = response
        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

        # Basic cleanup
        cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)  # Multiple newlines
        cleaned = cleaned.strip()

        return cleaned

    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process query with simple crew workflow"""
        try:
            logger.info("Processing query with simple crew", query=query[:100])

            # Task 1: Find documents
            find_task = Task(
                description=f"Find relevant documents for this question: {query}",
                agent=self.document_finder,
                expected_output="Relevant documents with content and scores"
            )

            # Check if this is a follow-up/formatting request
            task_description = f"Write a clear answer to this question using the documents: {query}"

            # If query has conversation context, it might be a follow-up
            if "CONVERSATION CONTEXT:" in query:
                task_description = f"""Analyze this query with conversation context: {query}

                If this is a formatting request (like "give in 3 points", "summarize", "list"),
                use the previous conversation to reformat the answer.

                If this is a new content question, search for new information."""

            # Task 2: Write answer
            answer_task = Task(
                description=task_description,
                agent=self.answer_writer,
                expected_output="Clear, factual answer based on the documents"
            )

            # Create and run crew
            crew = Crew(
                agents=[self.document_finder, self.answer_writer],
                tasks=[find_task, answer_task],
                process=Process.sequential,
                verbose=False
            )

            # Execute
            result = crew.kickoff()

            # Extract response
            if hasattr(result, 'tasks_output') and result.tasks_output:
                raw_response = str(result.tasks_output[-1].raw)
            else:
                raw_response = str(result)

            # Minimal cleaning
            clean_response = self._clean_response(raw_response)

            # Use sources from the tool call instead of recalculating
            sources = []
            if hasattr(self, '_last_sources') and self._last_sources:
                for source_data in self._last_sources:
                    source_info = {
                        "content": source_data["content"][:200] + "..." if len(source_data["content"]) > 200 else source_data["content"],
                        "score": source_data["score"],
                        "metadata": source_data["metadata"]
                    }
                    sources.append(source_info)
                # Clear for next query
                self._last_sources = []
            else:
                logger.warning("No sources found from tool call")

            return {
                "response": clean_response,
                "sources": sources,
                "metadata": {
                    "processing_mode": "simple_crew",
                    "agents_used": ["document_finder", "answer_writer"]
                }
            }

        except Exception as e:
            logger.error("Simple crew processing failed", error=str(e))
            raise Exception(f"Processing failed: {str(e)}")