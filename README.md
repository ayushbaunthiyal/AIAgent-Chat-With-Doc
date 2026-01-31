# RAG Chat Assistant

A conversational AI assistant that answers questions about content from document collections using RAG (Retrieval-Augmented Generation) with LangGraph ReAct agent, MCP Server, and Chroma vector database.

## 🏗️ Architecture Overview

The system follows a simplified RAG architecture with:
- **LangGraph ReAct Agent**: Intelligent reasoning and tool calling
- **MCP Server**: Primary context source for document retrieval
- **Chroma Vector Store**: Persistent storage for embeddings, chunks, and metadata
- **Streamlit UI**: Web interface for document upload and chat

```
User Query → Streamlit UI → LangGraph ReAct Agent → MCP Server Tools → Chroma Vector Store
                                                          ↓
                                                    OpenAI LLM → Response
```

## 🚀 Quick Setup

### Prerequisites

- Python 3.10-3.13 (3.13 recommended)
- Docker Desktop (for containerized deployment)
- OpenAI API key
- UV package manager (optional, but recommended)

### Installation

#### Option 1: Automated Setup (Recommended)

**Windows (PowerShell):**
```powershell
.\setup.ps1
```

**Linux/Mac:**
```bash
chmod +x setup.sh
./setup.sh
```

**Cross-platform (Python):**
```bash
python setup.py
```

The setup script will:
- ✅ Check Python version (3.10-3.13, 3.13 recommended)
- ✅ Install UV package manager (optional)
- ✅ Create necessary directories
- ✅ Create .env file from template
- ✅ Install all dependencies
- ✅ Verify installation

#### Option 2: Manual Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd AIAgent-Chat-With-Doc
   ```

2. **Install dependencies using UV**
   ```bash
   # Install UV if not already installed
   pip install uv

   # Install project dependencies
   uv pip install -e .
   ```

   Or using pip:
   ```bash
   pip install -e .
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

   Access the app at `http://localhost:8501`

### Docker Deployment

1. **Build the Docker image**
   ```bash
   docker build -t rag-chat-assistant .
   ```

2. **Run the container**
   ```bash
   docker run -p 8501:8501 \
     -v $(pwd)/data:/app/data \
     -v $(pwd)/.env:/app/.env \
     rag-chat-assistant
   ```

   On Windows PowerShell:
   ```powershell
   docker run -p 8501:8501 `
     -v ${PWD}/data:/app/data `
     -v ${PWD}/.env:/app/.env `
     rag-chat-assistant
   ```

## 📋 Features

- **Document Upload**: Support for PDF, TXT, and MD files
- **Intelligent Chunking**: Recursive character splitting with overlap (1000 chars, 200 overlap)
- **Semantic Search**: Vector-based similarity search using Chroma
- **ReAct Agent**: LangGraph-powered agent with reasoning capabilities
- **MCP Integration**: Tool-based document access via MCP Server
- **Persistent Storage**: Chroma vector store with local persistence
- **Short-term Memory**: In-memory conversation history (session-based)

## 🔧 Configuration

Key configuration options in `.env`:

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `OPENAI_MODEL`: LLM model (default: `gpt-4-turbo-preview`)
- `OPENAI_EMBEDDING_MODEL`: Embedding model (default: `text-embedding-3-small`)
- `CHROMA_DB_PATH`: Path to Chroma database (default: `./data/chroma_db`)
- `CHUNK_SIZE`: Chunk size in characters (default: `1000`)
- `CHUNK_OVERLAP`: Overlap between chunks (default: `200`)
- `TOP_K_CHUNKS`: Number of chunks to retrieve (default: `5`)
- `MAX_ITERATIONS`: Max ReAct iterations (default: `10`)

## 🧩 Tech Stack

- **Python 3.10-3.13**: Programming language (3.13 recommended)
- **UV**: Fast Python package manager
- **OpenAI**: LLM and embeddings
- **LangGraph**: Agent orchestration framework
- **LangChain**: LLM framework and MCP adapters
- **Chroma**: Vector database (local persistent mode)
- **Streamlit**: Web UI framework
- **MCP**: Model Context Protocol for tool integration
- **Docker**: Containerization

## 📁 Project Structure

```
AIAgent-Chat-With-Doc/
├── src/
│   ├── __init__.py
│   ├── config.py                 # Configuration management
│   ├── document_processor.py     # PDF/text loading and chunking
│   ├── embeddings.py             # Embedding generation service
│   ├── vector_store.py           # Chroma vector store wrapper
│   ├── mcp_client.py             # MCP Server client
│   ├── mcp_server/               # Local MCP Server
│   │   ├── __init__.py
│   │   ├── server.py             # MCP Server setup
│   │   └── tools.py              # MCP tools for document operations
│   ├── retrieval.py              # Hybrid retrieval service
│   ├── agent.py                  # LangGraph ReAct agent
│   ├── prompts.py                # Prompt templates
│   └── utils.py                  # Helper functions
├── app.py                        # Streamlit main application
├── data/                         # Data directory (gitignored)
│   └── chroma_db/               # Chroma persistent storage
├── tests/                        # Unit tests
├── pyproject.toml                # UV package configuration
├── Dockerfile                    # Docker container definition
├── .env.example                  # Environment variables template
└── README.md                     # This file
```

## 🔍 RAG/LLM Approach & Decisions

### LLM Selection
- **Primary**: GPT-4-turbo (high quality, good reasoning)
- **Fallback**: GPT-3.5-turbo (cost-effective alternative)
- **Rationale**: GPT-4 provides better reasoning for ReAct agent, GPT-3.5 for cost optimization

### Embedding Model
- **Model**: `text-embedding-3-small` (1536 dimensions)
- **Rationale**: Cost-effective, good quality, sufficient dimensions for semantic search

### Vector Database
- **Choice**: Chroma (local persistent mode)
- **Rationale**: 
  - Zero infrastructure costs (local file storage)
  - Easy setup, no external dependencies
  - Persistent storage (survives restarts)
  - Good performance for local use cases

### Orchestration Framework
- **Choice**: LangGraph
- **Rationale**:
  - Built-in ReAct pattern support
  - State management
  - Tool integration
  - Extensible workflow

### Chunking Strategy
- **Method**: Recursive character splitting with overlap
- **Size**: 1000 characters
- **Overlap**: 200 characters
- **Rationale**: Balances context preservation with retrieval granularity

### Retrieval Approach
- **Primary**: MCP Server tools (structured, tool-based access)
- **Fallback**: Direct Chroma vector search (semantic similarity)
- **Top-k**: 5 chunks per query
- **Re-ranking**: Relevance threshold filtering (>0.7)

### Prompt Engineering
- **ReAct System Prompt**: Instructions for reasoning, tool usage, and decision-making
- **Final Response Prompt**: Context assembly with source citations
- **Techniques**: Chain-of-thought reasoning, tool selection guidance, context synthesis

### Guardrails & Quality
- Input sanitization (prevent prompt injection)
- Response validation (ensure answers reference documents)
- Token limits (prevent context overflow)
- Relevance filtering (threshold-based)
- Error handling and fallbacks

### Observability
- Structured logging (query, retrieval, response)
- Performance metrics (latency tracking)
- Error tracking with context

## 🚢 Production Considerations

### Current State (Local Deployment)
- ✅ Local Docker Desktop deployment
- ✅ Persistent Chroma storage
- ✅ Environment-based configuration
- ✅ Basic logging and error handling

### What Would Be Required for Production

#### Scalability
- **Horizontal Scaling**: Multiple app instances with shared Chroma directory (or move to cloud vector DB)
- **Caching Layer**: Redis for frequent queries (can run in Docker locally)
- **Load Balancing**: Nginx or similar (can run in Docker)
- **Async Processing**: Background tasks for document ingestion

#### Deployment (Cloud)
- **Container Orchestration**: Kubernetes/ECS for multi-instance deployment
- **Cloud Vector DB**: Pinecone/Weaviate for distributed access
- **Managed Services**: Use cloud-managed databases and services
- **CI/CD Pipeline**: Automated testing and deployment

#### Monitoring
- **APM**: Application Performance Monitoring (e.g., Datadog, New Relic)
- **LLM Cost Tracking**: Monitor OpenAI API usage and costs
- **User Analytics**: Track usage patterns and queries
- **Alerting**: Set up alerts for errors and performance issues

#### Security
- **API Key Management**: AWS Secrets Manager, Azure Key Vault, or similar
- **Authentication**: User authentication/authorization
- **Data Encryption**: Encryption at rest and in transit
- **Input Validation**: Enhanced sanitization and validation
- **Rate Limiting**: Per-user rate limits

#### Quality & Testing
- **Integration Tests**: End-to-end testing
- **Performance Tests**: Load testing and benchmarking
- **Quality Metrics**: RAG evaluation metrics (retrieval accuracy, answer quality)
- **A/B Testing**: Compare different retrieval strategies

## 🧪 Testing

Run tests with:
```bash
pytest tests/
```

With coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

## 🛠️ Engineering Standards

- **Code Quality**: Type hints, docstrings, PEP 8 compliance
- **Package Management**: UV with `pyproject.toml`
- **Testing**: Unit tests for core logic (target 70%+ coverage)
- **Documentation**: Inline comments, README, architecture docs
- **Version Control**: Clear commit messages, feature branches
- **Dependencies**: Pinned versions in `pyproject.toml`
- **Containerization**: Multi-stage Docker builds

## 🤖 AI Tool Usage

### How AI Tools Were Used
- **Boilerplate Code**: Generated initial project structure and configuration files
- **Documentation Templates**: Created README and docstring templates
- **Code Generation**: Assisted with LangGraph and MCP integration patterns
- **Debugging Help**: Troubleshooting integration issues

### Do's and Don'ts

**Do's:**
- ✅ Use AI for repetitive tasks and boilerplate
- ✅ Review all AI-generated code before committing
- ✅ Test thoroughly, especially integrations
- ✅ Document AI-assisted sections in comments
- ✅ Understand the code you're using

**Don'ts:**
- ❌ Don't blindly accept AI suggestions
- ❌ Don't use AI for critical architectural decisions without review
- ❌ Don't skip testing AI-generated code
- ❌ Don't use AI outputs directly in README (write your own thoughts)

## 🔮 Future Enhancements

With more time, I would add:

1. **Enhanced MCP Integration**: Full tool binding and async tool calling
2. **Advanced ReAct Loop**: More sophisticated reasoning and multi-step planning
3. **Conversation Memory**: Persistent conversation history (SQLite or similar)
4. **Document Management**: Delete, update, and organize documents
5. **Multi-document Queries**: Cross-document reasoning
6. **Citation Tracking**: Better source attribution and references
7. **Export Functionality**: Export conversations and document summaries
8. **Performance Optimization**: Caching, batch processing, async operations
9. **Quality Metrics**: RAG evaluation and quality scoring
10. **User Feedback**: Thumbs up/down for continuous improvement

## 📝 License

[Add your license here]

## 👤 Author

[Your name/contact information]
