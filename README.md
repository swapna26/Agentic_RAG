# Agentic RAG Backend

Production-ready FastAPI backend for Retrieval-Augmented Generation with integrated conversation memory and CrewAI agents. Features OpenWebUI compatibility for seamless chat interface integration and intelligent context-aware responses.

## Processing Flow

The system processes user queries using CrewAI multi-agent system for intelligent document retrieval and response generation.

Context handling:
- Conversation history is provided to agents for context-aware responses
- Agents intelligently determine relevance and handle topic transitions
- Clean separation between new topics and follow-up questions

## Endpoints (OpenAI compatible)

- GET `/api/models`
- POST `/api/chat/completions`
- GET `/health`
- GET `/info`

## Configuration (env)

- `DATABASE_URL=postgresql://raguser:ragpassword@localhost:5432/agentic_rag`
- `OLLAMA_BASE_URL=http://localhost:11434`
- `OLLAMA_MODEL=llama3.2:1b`
- `OLLAMA_EMBEDDING_MODEL=nomic-embed-text:v1.5`
- `SIMILARITY_TOP_K=5`
- `TEMPERATURE=0.1`
- `MODEL_NAME=agentic-rag-ollama`
- `CREW_VERBOSE=true`

## Run

### Local

```
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Docker Compose

```
docker compose --profile backend up -d
# Services: postgres, ollama, rag_backend, openwebui
```

## OpenWebUI


- Model listed as: `agentic-rag-ollama`
- Chat via `/api/chat/completions`

## CrewAI Behavior (output hygiene)

- Uses configured Ollama model: `ollama/<OLLAMA_MODEL>`
- Plain-text outputs: headings/bold/lists removed
- Inline “Sources” removed from answer (sources returned separately)
- No embedded follow-up questions or planner artifacts

## Evaluation (RAGas)

**Current Setup (Ollama Llama3.2:1b):**
Run in 3 batches to avoid timeouts:
1) `answer_similarity`, `context_recall`
2) `faithfulness`, `answer_relevancy`
3) `context_precision`, `answer_correctness`

Merge the three reports into one combined JSON.

**Evaluation Challenges:**
- Some LLM-heavy metrics (relevancy, precision, correctness) face compatibility issues with Llama3.2:1b on Ollama
- Current model has limitations with complex evaluation tasks
- Timeout issues require batched evaluation approach

**Scalability Potential:**
With better infrastructure and larger LLM models (e.g., Llama3.2:7b+), significant accuracy improvements would be achievable across all RAGas metrics without timeout constraints.

## Troubleshooting (current)

**Model & Infrastructure:**
- Ensure models are pulled: `ollama pull llama3.2:1b` and `ollama pull nomic-embed-text:v1.5`
- Llama3.2:1b chosen for resource efficiency over larger models (3b, 7b, 70b)

**RAGas Evaluation:**
- LLM-heavy metrics (relevancy, precision, correctness) may timeout on Ollama
- Some metrics have compatibility issues with local Ollama setup
- Use batched evaluation and merge results as workaround
- Production deployments with cloud LLMs (GPT-4, Claude) would eliminate these constraints

## Architecture Overview

The backend implements an intelligent chatbot system with conversation memory and multi-agent RAG processing:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   OpenWebUI     │    │   FastAPI        │    │   PostgreSQL    │
│   Chat Client   │◄──►│   Backend        │◄──►│   + PGVector    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │    CrewAI        │
                       │  Multi-Agent     │
                       │   Processing     │
                       └──────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   Ollama LLM     │
                       │   + Embeddings   │
                       └──────────────────┘
```

## Key Features

- **CrewAI Multi-Agent Processing**: Intelligent processing with specialized retrieval and response agents
- **Conversation Memory**: Persistent context management with PostgreSQL storage
- **OpenWebUI Integration**: Seamless chat interface with OpenAI API compatibility
- **Context-Aware Responses**: Follow-up questions answered with full conversation context
- **Production Ready**: Comprehensive error handling, logging, and monitoring

## Core Components

### CrewAI Agents (`agents/crew_agents.py`)

**Primary Processing System** with specialized agent roles:

**Agent Team:**
- **Document Retrieval Specialist** - Finds relevant documents from the knowledge base using domain-aware search
- **Answer Generator** - Writes helpful answers using retrieved documents and conversation context

**Processing Flow:**
1. **Document Retrieval** - Agent searches vector store for relevant documents using domain-aware keywords
2. **Response Generation** - Agent creates contextually appropriate answers from retrieved documents

### RAG Service (`services/rag_service.py`)

**CrewAI Multi-Agent Processing:**

- Intelligent multi-agent system for document analysis
- Context-aware conversation handling
- Specialized agents for retrieval and response generation
- Integrated with PostgreSQL for conversation memory
- Processing time: 10-30 seconds for comprehensive analysis

### Conversation Memory System

**Memory Components:**
- **PostgreSQL Storage**: Persistent conversation history across sessions
- **Context Management**: Automatic extraction and formatting from OpenWebUI messages
- **Agent Integration**: Conversation context provided to CrewAI agents for intelligent responses

**Memory Flow:**
```
OpenWebUI Messages → PostgreSQL Storage → Context Extraction → CrewAI Agents → Context-Aware Response
```

### Chat Router (`routers/chat.py`)

**OpenWebUI Compatible Endpoints:**
- **POST /api/chat/completions** - Main chat endpoint with conversation support
- **GET /api/models** - Model listing for OpenWebUI integration
- **GET /health** - System health check

**Features:**
- **Conversation History Extraction**: Automatically extracts message history from OpenWebUI format
- **Memory Integration**: Passes conversation context to RAG service
- **Error Handling**: Comprehensive error responses and fallback mechanisms

## Request Flow

### Complete Processing Flow

```
1. OpenWebUI Request
   │
   ▼
2. Extract Conversation History
   ├── Current Message: "Give me a summary"
   └── Previous Context: [{"role": "user", "content": "What is procurement?"}, ...]
   │
   ▼
3. Store Conversation Context
   └── PostgreSQL.store(conversation_history)
   │
   ▼
4. CrewAI Multi-Agent Processing
   ├── Document Retrieval Specialist: Search vector store for relevant documents
   └── Answer Generator: Create comprehensive answer from retrieved documents
   │
   ▼
5. Intelligent Response
   └── LLM automatically relates current question to previous context
```

### Processing Details

| Component | Function | Memory | Speed | Use Case |
|-----------|----------|--------|-------|----------|
| CrewAI Agents | Multi-agent processing | Full context | 10-30s | Intelligent document analysis and conversation |
| PostgreSQL | Conversation storage | Persistent | Fast | Context retrieval and history management |
| Vector Store | Document retrieval | Embedding-based | Fast | Semantic document search |

## Getting Started

### Prerequisites

- Docker and Docker Compose
- PostgreSQL with PGVector extension
- Ollama with required models

### Required Ollama Models

```bash
ollama pull llama3.2:1b              # Main LLM
ollama pull nomic-embed-text:v1.5    # Embedding model
```

### Environment Variables

Create `.env` file:

```env
# Database
DATABASE_URL=postgresql://raguser:ragpassword@localhost:5432/agentic_rag

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:1b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text:v1.5

# Server Settings
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=*

# RAG Settings
SIMILARITY_TOP_K=5
MAX_TOKENS=4000
TEMPERATURE=0.1

# Conversation Memory
CONVERSATION_TOKEN_LIMIT=2000
MEMORY_CLEANUP_INTERVAL=100

# CrewAI Settings
CREW_VERBOSE=true
CREW_MEMORY=false

# Phoenix Observability (Optional)
PHOENIX_BASE_URL=http://localhost:6006
PHOENIX_PROJECT_NAME=agentic_rag_backend
```

### Running the Backend

```bash
# Using Docker Compose
docker-compose up rag_backend

# Local Development
pip install -r requirements.txt
python main.py
```

## API Endpoints

### Chat Completions (OpenWebUI Compatible)

```bash
POST /api/chat/completions
```

**Conversation Example:**
```json
{
  "model": "agentic-rag-ollama",
  "messages": [
    {"role": "user", "content": "What is procurement?"},
    {"role": "assistant", "content": "Procurement is the process of..."},
    {"role": "user", "content": "Give me a 3-line summary"}
  ],
  "stream": false
}
```

**Response:**
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "agentic-rag-ollama",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "# Procurement Summary\n\n- Procurement is the process of acquiring goods and services\n- It involves planning, sourcing, and contract management\n- Key steps include requirement analysis, supplier selection, and negotiation\n\nSources\n1. Abu Dhabi Procurement Standards.PDF (relevance: 0.71)\n2. Procurement Manual (Business Process).PDF (relevance: 0.69)\n\nProcessed using: retrieval_specialist, response_generator, validator"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 150,
    "completion_tokens": 85,
    "total_tokens": 235
  }
}
```

### Model Information

```bash
GET /api/models
```

**Response:**
```json
{
  "object": "list",
  "data": [{
    "id": "agentic-rag-ollama",
    "object": "model",
    "created": 1234567890,
    "owned_by": "agentic-rag",
    "permission": [],
    "root": "agentic-rag-ollama",
    "parent": null
  }]
}
```

### Health Check

```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "agentic-rag-backend",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## Conversation Examples

### Example 1: Follow-up Questions

**User:** "What are the main procurement policies?"
**Assistant:** *CrewAI agents analyze the question, retrieve relevant documents, and provide detailed procurement policy information*

**User:** "Can you summarize that in 5 bullet points?"
**Assistant:** *Query Analyzer identifies this as a FOLLOW_UP question, Document Retrieval Specialist searches for procurement summary information, Information Extractor creates 5 bullet points*

### Example 2: Context Switching

**User:** "Tell me about data protection requirements"
**Assistant:** *CrewAI agents process the new topic and provide data protection information*

**User:** "How does this relate to procurement?"
**Assistant:** *Query Analyzer identifies the relationship question, agents search for connections between data protection and procurement*

### Example 3: Complex Analysis

**User:** "Compare the approval processes for different procurement amounts"
**Assistant:** *CrewAI agents perform complex analysis across multiple documents to compare approval processes for different procurement thresholds*

## Configuration

### CrewAI Settings

```env
# CrewAI Configuration
CREW_VERBOSE=true          # Enable detailed agent logging
CREW_MEMORY=false          # Disable CrewAI memory (use conversation memory instead)

# Agent Processing
MAX_ITER=3                 # Maximum agent iterations
MAX_EXECUTION_TIME=300     # Agent timeout (seconds)
```

### Memory Management

```env
# Conversation memory settings
MAX_TOKENS=4000                    # Total token limit
CONVERSATION_TOKEN_LIMIT=2000      # Reserve for conversation memory
MEMORY_CLEANUP_INTERVAL=100        # Clean memory after N conversations
```

### Performance Optimization

```python
# Fast responses (CrewAI only)
SIMILARITY_TOP_K=3
CONVERSATION_TOKEN_LIMIT=1000

# Balanced performance
SIMILARITY_TOP_K=5
CONVERSATION_TOKEN_LIMIT=2000

# Maximum context retention
SIMILARITY_TOP_K=10
CONVERSATION_TOKEN_LIMIT=3000
```

## Development

### Testing the System

```bash
# Test chat completions
curl -X POST http://localhost:8000/api/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "agentic-rag-ollama",
    "messages": [
      {"role": "user", "content": "What is procurement?"},
      {"role": "assistant", "content": "Procurement is..."},
      {"role": "user", "content": "Give me a brief summary"}
    ]
  }'

# Test model listing
curl http://localhost:8000/api/models

# Test health check
curl http://localhost:8000/health
```

### Custom Agent Integration

```python
from services.rag_service import RAGService

# Direct usage
rag_service = RAGService(config)
await rag_service.initialize()

# Chat with conversation history
conversation_history = [
    {"role": "user", "content": "Previous question"},
    {"role": "assistant", "content": "Previous response"}
]

result = await rag_service.chat(
    "Current question",
    conversation_history,
    "conversation_id"
)
```

## Monitoring & Observability

### Processing Metrics

```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "event": "query_processed",
  "conversation_id": "conv_123",
  "processing_mode": "crew_ai_primary",
  "agents_used": ["retrieval_specialist", "response_generator", "validator"],
  "response_time_ms": 15000,
  "source_count": 3,
  "memory_token_usage": 1250
}
```

### Agent Performance Tracking

- **Agent Success Rate**: Track which agents complete successfully
- **Processing Time**: Monitor agent execution times
- **Fallback Frequency**: Track when CrewAI falls back to other methods
- **Context Understanding**: Measure conversation context accuracy

## Troubleshooting

### Common Issues

**CrewAI Agents Not Responding:**
- Check Ollama service status
- Verify database connection
- Review agent timeout settings
- Check CREW_VERBOSE logs

**Memory Issues:**
- Reduce `CONVERSATION_TOKEN_LIMIT`
- Check PostgreSQL connection
- Monitor memory usage in logs

**Slow Responses:**
- CrewAI agents take 10-30 seconds (normal)
- Check `MAX_EXECUTION_TIME` setting
- Consider reducing `SIMILARITY_TOP_K`

### Debug Mode

```env
LOG_LEVEL=DEBUG
CREW_VERBOSE=true
MEMORY_DEBUG=true
```

## Key Features

1. **CrewAI Primary Processing**: Intelligent multi-agent system for complex queries
2. **Intelligent Fallback**: Graceful degradation through three processing tiers
3. **Conversation Memory**: Full context awareness for follow-up questions
4. **OpenWebUI Integration**: Seamless chat interface with OpenAI API compatibility
5. **Production Ready**: Comprehensive error handling, logging, and monitoring
6. **Clean Code**: Professional, symbol-free codebase ready for presentations

## Integration

### OpenWebUI Setup

1. **Add Backend URL**: `http://localhost:8000/api`
2. **API Key**: `dummy-key-for-agentic-rag`
3. **Select Model**: `agentic-rag-ollama`
4. **Start Conversation**: Natural back-and-forth chat with intelligent document processing

The system provides a true chatbot experience where you can have natural conversations with your documents, ask follow-up questions, and the AI will maintain context throughout the conversation using intelligent multi-agent processing.

## License

Part of the Agentic RAG System. See main project for license information.

---

**Built for intelligent document conversations with enterprise-grade multi-agent processing and comprehensive memory management**