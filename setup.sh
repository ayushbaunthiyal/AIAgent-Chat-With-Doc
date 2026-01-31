#!/bin/bash
# Bash setup script for RAG Chat Assistant (Linux/Mac)

echo "============================================================"
echo "🚀 RAG Chat Assistant - Setup Script (Bash)"
echo "============================================================"
echo ""

# Check Python version
echo "🐍 Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ Python is not installed or not in PATH."
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "✅ Python version: $PYTHON_VERSION"

# Check if version is 3.10 or higher
MAJOR_VERSION=$(echo $PYTHON_VERSION | cut -d. -f1)
MINOR_VERSION=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$MAJOR_VERSION" -lt 3 ] || ([ "$MAJOR_VERSION" -eq 3 ] && [ "$MINOR_VERSION" -lt 10 ]); then
    echo "❌ Python 3.10 or higher is required."
    exit 1
fi

# Check/Install UV
echo ""
echo "📦 Checking UV package manager..."
UV_INSTALLED=false

if command -v uv &> /dev/null; then
    UV_VERSION=$(uv --version 2>&1)
    echo "✅ UV is installed: $UV_VERSION"
    UV_INSTALLED=true
else
    echo "⚠️  UV is not installed."
    read -p "❓ Install UV now? (y/n): " install_uv
    if [ "$install_uv" = "y" ] || [ "$install_uv" = "Y" ]; then
        echo "📦 Installing UV..."
        if curl -LsSf https://astral.sh/uv/install.sh | sh; then
            echo "✅ UV installed successfully!"
            UV_INSTALLED=true
            # Add to PATH for current session
            export PATH="$HOME/.cargo/bin:$PATH"
        else
            echo "❌ Failed to install UV."
            echo "⚠️  Continuing with pip instead..."
        fi
    else
        echo "⚠️  Continuing with pip instead..."
    fi
fi

# Create directories
echo ""
echo "📁 Creating directories..."
directories=("data" "data/chroma_db" "tests")

for dir in "${directories[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo "   ✅ Created: $dir"
    else
        echo "   ℹ️  Already exists: $dir"
    fi
done

# Create .env file
echo ""
echo "🔧 Setting up environment file..."
if [ -f ".env" ]; then
    echo "   ℹ️  .env file already exists, skipping..."
else
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "   ✅ Created .env from .env.example"
    else
        # Create basic .env file
        cat > .env << 'EOF'
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4-turbo-preview
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Chroma Configuration
CHROMA_DB_PATH=./data/chroma_db
CHROMA_COLLECTION_NAME=documents

# MCP Server Configuration
MCP_SERVER_TRANSPORT=stdio
MCP_SERVER_COMMAND=python
MCP_SERVER_ARGS=-m src.mcp_server.server

# Application Configuration
STREAMLIT_PORT=8501
STREAMLIT_HOST=0.0.0.0
LOG_LEVEL=INFO

# Chunking Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Retrieval Configuration
TOP_K_CHUNKS=5
RELEVANCE_THRESHOLD=0.7

# ReAct Agent Configuration
MAX_ITERATIONS=10
TEMPERATURE=0.7
EOF
        echo "   ✅ Created .env file"
    fi
    echo "   ⚠️  Please edit .env and add your OpenAI API key!"
fi

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
if [ "$UV_INSTALLED" = true ]; then
    echo "   Using UV package manager..."
    uv pip install -e .
else
    echo "   Using pip..."
    $PYTHON_CMD -m pip install -e .
fi

if [ $? -eq 0 ]; then
    echo "   ✅ Dependencies installed successfully!"
else
    echo "   ❌ Failed to install dependencies."
    exit 1
fi

# Verify installation
echo ""
echo "🔍 Verifying installation..."
required_packages=("openai" "langchain" "langgraph" "chromadb" "streamlit" "pypdf")
missing_packages=()

for package in "${required_packages[@]}"; do
    package_name=$(echo $package | tr '-' '_')
    if $PYTHON_CMD -c "import $package_name" 2>/dev/null; then
        echo "   ✅ $package"
    else
        echo "   ❌ $package (missing)"
        missing_packages+=("$package")
    fi
done

if [ ${#missing_packages[@]} -gt 0 ]; then
    echo ""
    echo "⚠️  Missing packages: ${missing_packages[*]}"
    echo "   Please install them manually: pip install ${missing_packages[*]}"
else
    echo ""
    echo "✅ All required packages are installed!"
fi

# Final instructions
echo ""
echo "============================================================"
echo "✅ Setup completed successfully!"
echo "============================================================"
echo ""
echo "📝 Next steps:"
echo "   1. Edit .env file and add your OpenAI API key"
echo "   2. Run the application:"
echo "      streamlit run app.py"
echo ""
echo "🐳 Or use Docker:"
echo "   docker build -t rag-chat-assistant ."
echo "   docker run -p 8501:8501 -v \$(pwd)/data:/app/data rag-chat-assistant"
echo ""
echo "============================================================"
