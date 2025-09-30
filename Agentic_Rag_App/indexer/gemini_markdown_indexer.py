"""Gemini-based markdown indexer for better embeddings and text splitting."""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import hashlib

import click
import structlog
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.logging import RichHandler

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter, MarkdownNodeParser
from llama_index.core.schema import Document, TextNode
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding

from config import config

# Configure logging
logging.basicConfig(
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
    level=getattr(logging, config.log_level.upper(), logging.INFO),
)

logger = structlog.get_logger()
console = Console()


class GeminiMarkdownIndexer:
    """Gemini-based indexer for markdown files with improved text splitting."""

    def __init__(self):
        self.config = config
        self.vector_store = None
        self.embed_model = None
        self.index = None

        # Document tracking
        self.processed_docs = {}
        self.failed_docs = []

    async def initialize(self) -> bool:
        """Initialize the Gemini embedding model and vector store."""
        try:
            # Configure embedding model based on provider
            if self.config.llm_provider == 'gemini':
                logger.info("Initializing Gemini embedding model", model=self.config.gemini_embedding_model)
                self.embed_model = GeminiEmbedding(
                    model_name=self.config.gemini_embedding_model,
                    api_key=self.config.gemini_api_key,
                )
            else:
                logger.info("Initializing Ollama embedding model", model=self.config.ollama_embedding_model)
                self.embed_model = OllamaEmbedding(
                    model_name=self.config.ollama_embedding_model,
                    base_url=self.config.ollama_base_url,
                )

            # Initialize vector store
            self.vector_store = PGVectorStore.from_params(
                database=self.config.database_url.split('/')[-1],
                host=self.config.database_url.split('@')[1].split(':')[0],
                password=self.config.database_url.split(':')[2].split('@')[0],
                port=int(self.config.database_url.split(':')[3].split('/')[0]),
                user=self.config.database_url.split('://')[1].split(':')[0],
                table_name="embeddings_gemini",  # Different table for Gemini embeddings
                embed_dim=768,  # Gemini text-embedding-004 dimension
            )

            # Configure global settings
            Settings.embed_model = self.embed_model
            Settings.chunk_size = self.config.chunk_size
            Settings.chunk_overlap = self.config.chunk_overlap

            # Initialize index
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store
            )

            logger.info("Gemini markdown indexer initialized successfully")
            return True

        except Exception as e:
            logger.error("Failed to initialize Gemini indexer", error=str(e))
            return False

    def _create_improved_node_parser(self) -> MarkdownNodeParser:
        """Create an improved node parser for better text splitting."""
        # Use MarkdownNodeParser for better markdown understanding
        markdown_parser = MarkdownNodeParser()

        # Create sentence splitter for refined chunking
        if tiktoken:
            sentence_splitter = SentenceSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                paragraph_separator="\n\n\n",
                secondary_chunking_regex="[^,.;。？！]+[,.;。？！]?",
                tokenizer=tiktoken.get_encoding("cl100k_base").encode
            )
        else:
            sentence_splitter = SentenceSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                paragraph_separator="\n\n\n",
                secondary_chunking_regex="[^,.;。？！]+[,.;。？！]?"
            )

        # Chain the parsers: markdown -> sentences
        return sentence_splitter

    def _load_markdown_files(self, markdown_path: str) -> List[Document]:
        """Load all markdown files from the specified directory."""
        markdown_dir = Path(markdown_path)

        if not markdown_dir.exists():
            logger.error("Markdown directory does not exist", path=str(markdown_dir))
            return []

        markdown_files = list(markdown_dir.glob("*.md"))

        if not markdown_files:
            logger.warning("No markdown files found", path=str(markdown_dir))
            return []

        documents = []

        for md_file in markdown_files:
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Create document with metadata
                doc = Document(
                    text=content,
                    metadata={
                        'filename': md_file.name,
                        'file_path': str(md_file),
                        'file_size': md_file.stat().st_size,
                        'source_type': 'markdown',
                        'processed_with': f'{self.config.llm_provider}_embeddings',
                        'chunk_strategy': 'improved_markdown_splitting'
                    }
                )

                documents.append(doc)
                logger.info("Loaded markdown file", filename=md_file.name, size_kb=round(md_file.stat().st_size/1024, 1))

            except Exception as e:
                logger.error("Failed to load markdown file", filename=md_file.name, error=str(e))
                self.failed_docs.append(str(md_file))
                continue

        return documents

    def _calculate_doc_hash(self, doc: Document) -> str:
        """Calculate hash for document to track changes."""
        content = f"{doc.text}{doc.metadata.get('filename', '')}"
        return hashlib.md5(content.encode()).hexdigest()

    async def _process_documents_batch(self, documents: List[Document]) -> int:
        """Process a batch of documents with improved chunking."""
        if not documents:
            return 0

        processed_count = 0

        try:
            # Create improved node parser
            node_parser = self._create_improved_node_parser()

            # Parse documents into nodes
            all_nodes = []
            for doc in documents:
                try:
                    # Calculate document hash
                    doc_hash = self._calculate_doc_hash(doc)

                    # Parse document into nodes
                    nodes = node_parser.get_nodes_from_documents([doc])

                    # Add document hash to each node
                    for node in nodes:
                        node.metadata.update(doc.metadata)
                        node.metadata['doc_hash'] = doc_hash
                        node.metadata['chunk_id'] = f"{doc.metadata['filename']}_{len(all_nodes)}"

                    all_nodes.extend(nodes)
                    processed_count += 1

                    logger.info(
                        "Processed document into chunks",
                        filename=doc.metadata['filename'],
                        chunks_created=len(nodes),
                        total_chars=len(doc.text)
                    )

                except Exception as e:
                    logger.error("Failed to process document", filename=doc.metadata.get('filename', 'unknown'), error=str(e))
                    continue

            # Insert nodes into vector store
            if all_nodes:
                logger.info("Inserting nodes into vector store", total_nodes=len(all_nodes))
                self.index.insert_nodes(all_nodes)
                logger.info("Successfully inserted nodes into vector store")

        except Exception as e:
            logger.error("Failed to process document batch", error=str(e))
            return 0

        return processed_count

    async def index_markdown_files(self, markdown_path: Optional[str] = None) -> int:
        """Index markdown files using Gemini embeddings."""
        # Use provided path or default from config
        md_path = markdown_path or self.config.markdown_path

        # Convert relative path to absolute
        if not os.path.isabs(md_path):
            md_path = str(Path(__file__).parent / md_path)

        console.print(f"Loading markdown files from: {md_path}", style="blue")

        # Load all markdown documents
        documents = self._load_markdown_files(md_path)

        if not documents:
            logger.warning("No markdown documents to process")
            return 0

        total_docs = len(documents)
        logger.info("Found markdown files to index", count=total_docs)

        # Display file summary
        console.print(f"\nFound {total_docs} markdown files:", style="green")
        for doc in documents:
            size_kb = doc.metadata['file_size'] / 1024
            console.print(f"  • {doc.metadata['filename']} ({size_kb:.1f} KB)")

        processed_count = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:

            task = progress.add_task(
                f"Indexing with {self.config.llm_provider.title()} embeddings...",
                total=total_docs
            )

            # Process documents in batches
            for i in range(0, total_docs, self.config.batch_size):
                batch = documents[i:i + self.config.batch_size]

                try:
                    batch_processed = await self._process_documents_batch(batch)
                    processed_count += batch_processed

                    progress.update(task, advance=len(batch))

                    batch_num = i // self.config.batch_size + 1
                    console.print(
                        f"Batch {batch_num}: {batch_processed}/{len(batch)} documents processed",
                        style="cyan"
                    )

                except Exception as e:
                    logger.error("Failed to process batch", batch_num=i//self.config.batch_size + 1, error=str(e))
                    progress.update(task, advance=len(batch))
                    continue

        # Show results
        console.print(f"\n✓ Indexing completed!", style="bold green")
        console.print(f"  • Total files: {total_docs}")
        console.print(f"  • Successfully processed: {processed_count}")
        console.print(f"  • Failed: {len(self.failed_docs)}")
        console.print(f"  • Embedding model: {self.config.llm_provider} ({self.config.gemini_embedding_model if self.config.llm_provider == 'gemini' else self.config.ollama_embedding_model})")
        console.print(f"  • Vector store table: embeddings_gemini")

        if self.failed_docs:
            console.print(f"\nFailed files:", style="red")
            for failed in self.failed_docs:
                console.print(f"  • {failed}")

        return processed_count

    async def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up Gemini indexer resources")


# Import tiktoken for tokenizer
try:
    import tiktoken
except ImportError:
    logger.warning("tiktoken not available, using default tokenizer")
    tiktoken = None


@click.group()
def cli():
    """Gemini Markdown Indexer for Agentic RAG System."""
    pass


@cli.command()
@click.option('--path', '-p', help='Path to markdown files directory')
@click.option('--provider', '-pr', default='gemini', help='Embedding provider (gemini/ollama)')
def index(path: Optional[str], provider: str):
    """Index markdown files using Gemini embeddings."""
    async def run_indexing():
        # Override provider if specified
        if provider:
            config.llm_provider = provider

        indexer = GeminiMarkdownIndexer()

        try:
            console.print(f"🚀 Starting Gemini Markdown Indexer with {provider} embeddings...", style="bold blue")

            # Initialize
            if not await indexer.initialize():
                console.print("❌ Failed to initialize indexer", style="red")
                sys.exit(1)

            console.print("✓ Indexer initialized successfully", style="green")

            # Index markdown files
            count = await indexer.index_markdown_files(path)

            if count > 0:
                console.print(f"🎉 Successfully indexed {count} markdown files!", style="bold green")
            else:
                console.print("⚠️  No markdown files were indexed", style="yellow")

        except KeyboardInterrupt:
            console.print("⚠️  Indexing interrupted by user", style="red")
        except Exception as e:
            logger.error("Indexing failed", error=str(e))
            console.print(f"❌ Indexing failed: {e}", style="red")
            sys.exit(1)
        finally:
            await indexer.cleanup()

    asyncio.run(run_indexing())


@cli.command()
def status():
    """Check the status of the Gemini indexing system."""
    async def check_status():
        indexer = GeminiMarkdownIndexer()

        try:
            success = await indexer.initialize()
            if success:
                console.print("✓ Gemini indexer system is healthy", style="green")
                console.print(f"  • Provider: {config.llm_provider}")
                console.print(f"  • Embedding model: {config.gemini_embedding_model if config.llm_provider == 'gemini' else config.ollama_embedding_model}")
                console.print(f"  • Database: {config.database_url.split('@')[1]}")
            else:
                console.print("❌ Gemini indexer system is not healthy", style="red")
                sys.exit(1)

        except Exception as e:
            console.print(f"❌ Status check failed: {e}", style="red")
            sys.exit(1)
        finally:
            await indexer.cleanup()

    asyncio.run(check_status())


if __name__ == '__main__':
    cli()