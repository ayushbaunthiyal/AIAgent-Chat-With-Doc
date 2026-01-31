# RAG Chat Assistant

A conversational AI assistant that answers questions about your documents using **RAG (Retrieval-Augmented Generation)** with a **LangGraph ReAct agent**, **MCP Server**, and **Chroma vector database**.

![Python](https://img.shields.io/badge/Python-3.10--3.13-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-ReAct_Agent-green)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4_Turbo-orange)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Technical Design](#-technical-design)
- [Quick Start - Local Setup](#-quick-start---local-setup)
- [Quick Start - Docker Setup](#-quick-start---docker-setup)
- [Configuration](#-configuration)
- [Project Structure](#-project-structure)
- [RAG/LLM Approach & Decisions](#-ragllm-approach--decisions)
- [Production Considerations](#-production-considerations)
- [Testing](#-testing)
- [Troubleshooting](#-troubleshooting)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Document Upload** | Support for PDF, TXT, and Markdown files |
| **Intelligent Chunking** | Recursive character splitting (1000 chars, 200 overlap) |
| **Semantic Search** | Vector-based similarity search using Chroma |
| **ReAct Agent** | LangGraph-powered agent with multi-step reasoning |
| **MCP Integration** | Tool-based document access via Model Context Protocol |
| **Persistent Storage** | Chroma vector store with local file persistence |
| **Session Memory** | In-memory conversation history per session |
| **Collection Management** | Clear all documents with one click |
| **Source Citations** | Answers include references to source chunks |

---

## 🏗️ Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              STREAMLIT UI                                    │
│                         (Document Upload + Chat)                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          LANGGRAPH REACT AGENT                               │
│                                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │  THINK   │───▶│   ACT    │───▶│ OBSERVE  │───▶│ GENERATE │              │
│  │(Reason)  │    │(Use Tool)│    │(Process) │    │(Respond) │              │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘              │
│        ▲                                              │                      │
│        └──────────────────────────────────────────────┘                      │
│                      (Loop until answer ready)                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
┌───────────────────────────────┐     ┌───────────────────────────────────────┐
│        MCP SERVER             │     │            OPENAI API                  │
│  (Model Context Protocol)     │     │                                        │
│                               │     │  ┌─────────────┐  ┌─────────────────┐ │
│  Tools:                       │     │  │ GPT-4 Turbo │  │ text-embedding- │ │
│  • search_documents           │     │  │   (LLM)     │  │   3-small       │ │
│  • get_document_chunk         │     │  └─────────────┘  └─────────────────┘ │
│  • list_documents             │     │                                        │
└───────────────────────────────┘     └───────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CHROMA VECTOR DATABASE                               │
│                        (Local Persistent Storage)                            │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  Collection: "documents"                                             │   │
│   │  ├── Document Chunks (text content)                                  │   │
│   │  ├── Embeddings (1536-dim vectors)                                   │   │
│   │  └── Metadata (source, chunk_id, page_number)                        │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   Storage: ./data/chroma_db/ (SQLite + HNSW index)                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
1. DOCUMENT INGESTION
   Upload PDF/TXT → Chunking (1000 chars) → Embedding → Store in Chroma

2. QUERY PROCESSING
   User Query → ReAct Agent → MCP Tools → Chroma Search → Retrieve Chunks

3. RESPONSE GENERATION
   Retrieved Chunks → Context Assembly → GPT-4 → Answer with Citations
```

### Component Interactions

```
┌────────────────┐      ┌────────────────┐      ┌────────────────┐
│   Streamlit    │─────▶│  LangGraph     │─────▶│    OpenAI      │
│   (app.py)     │◀─────│  ReAct Agent   │◀─────│    LLM API     │
└────────────────┘      └────────────────┘      └────────────────┘
        │                       │
        │                       │ Tool Calls
        │                       ▼
        │               ┌────────────────┐
        │               │   MCP Server   │
        │               │   (stdio)      │
        │               └────────────────┘
        │                       │
        │                       ▼
        │               ┌────────────────┐
        └──────────────▶│    Chroma      │
        (Direct access) │  Vector Store  │
                        └────────────────┘
```

---

## 🔧 Technical Design

### Core Components

| Component | File | Purpose |
|-----------|------|---------|
| **Config** | `src/config.py` | Pydantic settings, environment variable management |
| **Document Processor** | `src/document_processor.py` | PDF/TXT loading, recursive chunking |
| **Embeddings** | `src/embeddings.py` | OpenAI embedding generation wrapper |
| **Vector Store** | `src/vector_store.py` | Chroma operations (add, search, delete) |
| **MCP Server** | `src/mcp_server/` | Tool definitions, server setup |
| **MCP Client** | `src/mcp_client.py` | MCP adapter for LangGraph |
| **Retrieval** | `src/retrieval.py` | Hybrid retrieval (MCP + vector search) |
| **Agent** | `src/agent.py` | LangGraph ReAct agent implementation |
| **Prompts** | `src/prompts.py` | System prompts and templates |
| **App** | `app.py` | Streamlit web interface |

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Agent Framework** | LangGraph ReAct | Built-in reasoning loop, state management, tool integration |
| **Vector Database** | Chroma (local) | Zero infrastructure cost, persistent, easy setup |
| **LLM** | GPT-4-turbo | Best reasoning for ReAct, high-quality responses |
| **Embeddings** | text-embedding-3-small | Cost-effective, 1536 dimensions, good quality |
| **Context Protocol** | MCP (stdio) | Standardized tool access, LangChain adapters available |
| **Memory** | Session-based (in-memory) | Simple, clears on refresh, no database needed |
| **Chunking** | RecursiveCharacterTextSplitter | Preserves context, configurable overlap |

### State Management

```python
# LangGraph Agent State
{
    "messages": List[BaseMessage],      # Conversation history (trimmed to last 10)
    "context": str,                      # Retrieved document chunks
    "iteration_count": int,              # ReAct loop counter
    "final_response": str                # Generated answer
}

# Streamlit Session State
{
    "messages": List[dict],              # Chat display history
    "agent": RAGAgent,                   # Agent instance
    "vector_store": VectorStore,         # Chroma wrapper
    "embedding_service": EmbeddingService,
    "document_processor": DocumentProcessor
}
```

### MCP Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `search_documents` | Semantic search across all documents | `query: str`, `top_k: int` |
| `get_document_chunk` | Retrieve specific chunk by ID | `chunk_id: str` |
| `list_documents` | List all stored document sources | None |

---

## 🚀 Quick Start - Local Setup

### Prerequisites

- **Python 3.10-3.13** (3.13 recommended)
- **OpenAI API Key** ([Get one here](https://platform.openai.com/api-keys))
- **Git** (for cloning)

### Step 1: Clone the Repository

```bash
git clone https://github.com/ayush-baunthiyal/AIAgent-Chat-With-Doc.git
cd AIAgent-Chat-With-Doc
```

### Step 2: Create Virtual Environment

**Option A: Using UV (Recommended - Faster)**
```bash
# Install UV if not already installed
pip install uv

# Create virtual environment
uv venv

# Activate virtual environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Windows CMD:
.\.venv\Scripts\activate.bat
# Linux/Mac:
source .venv/bin/activate
```

**Option B: Using standard Python venv**
```bash
python -m venv .venv

# Activate (same as above)
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux/Mac:
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Using UV (faster):
uv pip install -e .

# Or using pip:
pip install -e .
```

### Step 4: Configure Environment

```bash
# Create .env file from template
copy .env.example .env    # Windows
cp .env.example .env      # Linux/Mac

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-your-key-here
```

### Step 5: Run the Application

```bash
streamlit run app.py
```

**Open your browser: http://localhost:8501**

### Quick Setup Script (Alternative)

Instead of manual steps, use the automated setup:

```bash
# Windows PowerShell:
.\setup.ps1

# Linux/Mac:
chmod +x setup.sh && ./setup.sh

# Cross-platform Python:
python setup.py
```

---

## 🐳 Quick Start - Docker Setup

### Prerequisites

- **Docker Desktop** ([Download](https://www.docker.com/products/docker-desktop/))
- **OpenAI API Key**

### Step 1: Clone the Repository

```bash
git clone https://github.com/ayush-baunthiyal/AIAgent-Chat-With-Doc.git
cd AIAgent-Chat-With-Doc
```

### Step 2: Create Environment File

```bash
# Create .env file
copy .env.example .env    # Windows
cp .env.example .env      # Linux/Mac

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-your-key-here
```

### Step 3: Build Docker Image

```bash
docker build -t rag-chat-assistant .
```

### Step 4: Run Container

**Windows PowerShell:**
```powershell
docker run -d --name rag-chat -p 8501:8501 --env-file .env -v "${PWD}/data:/app/data" rag-chat-assistant
```

**Linux/Mac:**
```bash
docker run -d --name rag-chat -p 8501:8501 --env-file .env -v "$(pwd)/data:/app/data" rag-chat-assistant
```

**Open your browser: http://localhost:8501**

### Docker Commands Reference

| Command | Description |
|---------|-------------|
| `docker logs rag-chat` | View container logs |
| `docker logs -f rag-chat` | Follow logs in real-time |
| `docker stop rag-chat` | Stop the container |
| `docker start rag-chat` | Start stopped container |
| `docker rm rag-chat` | Remove container |
| `docker rm -f rag-chat` | Force remove running container |

### Rebuild After Code Changes

```bash
docker rm -f rag-chat
docker build -t rag-chat-assistant .
docker run -d --name rag-chat -p 8501:8501 --env-file .env -v "${PWD}/data:/app/data" rag-chat-assistant
```

---

## ⚙️ Configuration

### Environment Variables (.env)

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *required* | Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-4-turbo-preview` | LLM model for responses |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Model for embeddings |
| `CHROMA_DB_PATH` | `./data/chroma_db` | Path to Chroma database |
| `CHROMA_COLLECTION_NAME` | `documents` | Chroma collection name |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K_CHUNKS` | `5` | Chunks to retrieve per query |
| `RELEVANCE_THRESHOLD` | `0.3` | Minimum relevance score (0-1) |
| `MAX_ITERATIONS` | `10` | Max ReAct reasoning loops |
| `TEMPERATURE` | `0.7` | LLM creativity (0-1) |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

### Example .env File

```env
# Required
OPENAI_API_KEY=sk-your-api-key-here

# Optional - Model Configuration
OPENAI_MODEL=gpt-4-turbo-preview
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Optional - Retrieval Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_CHUNKS=5
RELEVANCE_THRESHOLD=0.3

# Optional - Agent Configuration
MAX_ITERATIONS=10
TEMPERATURE=0.7
```

---

## 📁 Project Structure

```
AIAgent-Chat-With-Doc/
├── app.py                      # Streamlit main application
├── src/
│   ├── __init__.py
│   ├── config.py               # Pydantic settings management
│   ├── document_processor.py   # PDF/text loading and chunking
│   ├── embeddings.py           # OpenAI embedding wrapper
│   ├── vector_store.py         # Chroma vector store operations
│   ├── mcp_client.py           # MCP adapter for LangGraph
│   ├── mcp_server/
│   │   ├── __init__.py
│   │   ├── server.py           # MCP server setup
│   │   └── tools.py            # Document search/retrieval tools
│   ├── retrieval.py            # Hybrid retrieval service
│   ├── agent.py                # LangGraph ReAct agent
│   ├── prompts.py              # System prompts and templates
│   └── utils.py                # Helper functions
├── data/
│   └── chroma_db/              # Chroma persistent storage
├── tests/
│   ├── __init__.py
│   └── test_document_processor.py
├── pyproject.toml              # Project dependencies (UV/pip)
├── Dockerfile                  # Container definition
├── setup.py                    # Cross-platform setup script
├── setup.ps1                   # Windows PowerShell setup
├── setup.sh                    # Linux/Mac setup
├── .env.example                # Environment template
├── .gitignore
└── README.md
```

---

## 🔍 RAG/LLM Approach & Decisions

### LLM Selection

| Model | Use Case | Rationale |
|-------|----------|-----------|
| **GPT-4-turbo** | Primary LLM | Best reasoning for ReAct agent, high-quality responses |
| **GPT-3.5-turbo** | Cost alternative | Lower cost, faster, acceptable for simple queries |

### Embedding Strategy

- **Model**: `text-embedding-3-small` (1536 dimensions)
- **Why**: Cost-effective ($0.02/1M tokens), good semantic quality
- **Alternative**: `text-embedding-3-large` for higher accuracy at 2x cost

### Chunking Strategy

```python
RecursiveCharacterTextSplitter(
    chunk_size=1000,      # ~250 words per chunk
    chunk_overlap=200,    # 20% overlap for context
    separators=["\n\n", "\n", ". ", " ", ""]
)
```

**Rationale**: Balances context preservation with retrieval granularity. Overlap prevents losing information at chunk boundaries.

### Retrieval Pipeline

1. **Query Embedding**: User query → 1536-dim vector
2. **Similarity Search**: Chroma HNSW index → top-k candidates
3. **Relevance Filtering**: Score threshold (>0.3) → filtered results
4. **Context Assembly**: Concatenate chunks with metadata

### ReAct Agent Loop

```
1. THINK: Analyze query, plan approach
2. ACT: Call MCP tools (search_documents, get_chunk)
3. OBSERVE: Process tool results
4. REPEAT: Until sufficient context gathered
5. GENERATE: Synthesize final answer with citations
```

### Prompt Engineering

- **System Prompt**: ReAct instructions, tool usage guidelines
- **Context Prompt**: Retrieved chunks with source metadata
- **Response Prompt**: Citation format, answer structure

---

## 🚢 Production Considerations

### Current State (Local Deployment)

✅ Local Docker Desktop deployment  
✅ Persistent Chroma storage  
✅ Environment-based configuration  
✅ Basic logging and error handling  
✅ Session-based conversation memory  
✅ Context length management (message trimming)  

### Scaling for Production

| Area | Local | Production |
|------|-------|------------|
| **Vector DB** | Chroma (local file) | Pinecone, Weaviate, Qdrant (cloud) |
| **LLM** | OpenAI API | Azure OpenAI, self-hosted LLM |
| **Memory** | Session (in-memory) | Redis, PostgreSQL |
| **Orchestration** | Single container | Kubernetes, ECS |
| **Monitoring** | Console logs | Datadog, New Relic, LangSmith |
| **Auth** | None | OAuth, API keys |

### Security Checklist

- [ ] API key management (secrets manager)
- [ ] Input sanitization (prompt injection prevention)
- [ ] Rate limiting per user
- [ ] Data encryption at rest
- [ ] HTTPS termination
- [ ] Authentication/authorization

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_document_processor.py -v
```

---

## 🔧 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| **PowerShell script disabled** | Run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| **Python 3.14+ errors** | Use Python 3.10-3.13 (onnxruntime compatibility) |
| **Docker context errors** | Run: `docker context use desktop-linux` |
| **localhost:8501 not loading** | Wait 10 seconds for Streamlit startup |
| **"Checkpointer requires thread_id"** | Update to latest code (MemorySaver removed) |
| **Context length exceeded** | Conversation is auto-trimmed (last 10 messages) |
| **No results found** | Lower `RELEVANCE_THRESHOLD` in .env |

### Logs

**Local:**
```bash
# Check Streamlit output in terminal
streamlit run app.py
```

**Docker:**
```bash
docker logs rag-chat
docker logs -f rag-chat  # Follow logs
```

---

## 🤖 AI Tool Usage

### How AI Was Used

- ✅ Boilerplate code generation
- ✅ Documentation templates
- ✅ LangGraph/MCP integration patterns
- ✅ Debugging assistance

### Best Practices

**Do:**
- Review all AI-generated code
- Test thoroughly
- Understand the code you're using

**Don't:**
- Blindly accept suggestions
- Skip testing
- Use AI for critical security code without review

---

## 📝 License

MIT License - Feel free to use and modify.

---

## 👤 Author

**Ayush Baunthiyal**  
Staff Software Engineer / AI Engineer

---

## 🔮 Future Enhancements

1. **Streaming responses** - Token-by-token display
2. **Multi-document reasoning** - Cross-document queries
3. **Persistent memory** - SQLite conversation history
4. **Document management** - Delete/update specific documents
5. **Export functionality** - Save conversations
6. **Quality metrics** - RAG evaluation scoring
7. **User feedback** - Thumbs up/down for improvement
