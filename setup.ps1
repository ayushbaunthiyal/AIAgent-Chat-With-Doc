# PowerShell setup script for RAG Chat Assistant

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "🚀 RAG Chat Assistant - Setup Script (PowerShell)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "🐍 Checking Python version..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -eq 0) {
    $version = $pythonVersion -replace "Python ", ""
    Write-Host "✅ Python version: $version" -ForegroundColor Green
    
    # Check if version is 3.10 or higher
    $majorVersion = [int]($version -split "\.")[0]
    $minorVersion = [int]($version -split "\.")[1]
    
    if ($majorVersion -lt 3 -or ($majorVersion -eq 3 -and $minorVersion -lt 10)) {
        Write-Host "❌ Python 3.10 or higher is required." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "❌ Python is not installed or not in PATH." -ForegroundColor Red
    exit 1
}

# Check/Install UV
Write-Host ""
Write-Host "📦 Checking UV package manager..." -ForegroundColor Yellow
$uvInstalled = $false
try {
    $uvVersion = uv --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ UV is installed: $uvVersion" -ForegroundColor Green
        $uvInstalled = $true
    }
} catch {
    Write-Host "⚠️  UV is not installed." -ForegroundColor Yellow
}

if (-not $uvInstalled) {
    $installUv = Read-Host "❓ Install UV now? (y/n)"
    if ($installUv -eq "y" -or $installUv -eq "Y") {
        Write-Host "📦 Installing UV..." -ForegroundColor Yellow
        try {
            Invoke-WebRequest -Uri "https://astral.sh/uv/install.ps1" -UseBasicParsing | Invoke-Expression
            Write-Host "✅ UV installed successfully!" -ForegroundColor Green
            $uvInstalled = $true
        } catch {
            Write-Host "❌ Failed to install UV: $_" -ForegroundColor Red
            Write-Host "⚠️  Continuing with pip instead..." -ForegroundColor Yellow
        }
    } else {
        Write-Host "⚠️  Continuing with pip instead..." -ForegroundColor Yellow
    }
}

# Create directories
Write-Host ""
Write-Host "📁 Creating directories..." -ForegroundColor Yellow
$directories = @("data", "data\chroma_db", "tests")

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "   ✅ Created: $dir" -ForegroundColor Green
    } else {
        Write-Host "   ℹ️  Already exists: $dir" -ForegroundColor Gray
    }
}

# Create .env file
Write-Host ""
Write-Host "🔧 Setting up environment file..." -ForegroundColor Yellow
if (Test-Path ".env") {
    Write-Host "   ℹ️  .env file already exists, skipping..." -ForegroundColor Gray
} else {
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-Host "   ✅ Created .env from .env.example" -ForegroundColor Green
    } else {
        # Create basic .env file
        $envContent = @"
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
"@
        Set-Content -Path ".env" -Value $envContent
        Write-Host "   ✅ Created .env file" -ForegroundColor Green
    }
    Write-Host "   ⚠️  Please edit .env and add your OpenAI API key!" -ForegroundColor Yellow
}

# Install dependencies
Write-Host ""
Write-Host "📦 Installing dependencies..." -ForegroundColor Yellow
try {
    if ($uvInstalled) {
        Write-Host "   Using UV package manager..." -ForegroundColor Gray
        uv pip install -e .
    } else {
        Write-Host "   Using pip..." -ForegroundColor Gray
        python -m pip install -e .
    }
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   ✅ Dependencies installed successfully!" -ForegroundColor Green
    } else {
        Write-Host "   ❌ Failed to install dependencies." -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "   ❌ Error installing dependencies: $_" -ForegroundColor Red
    exit 1
}

# Verify installation
Write-Host ""
Write-Host "🔍 Verifying installation..." -ForegroundColor Yellow
$requiredPackages = @("openai", "langchain", "langgraph", "chromadb", "streamlit", "pypdf")
$missingPackages = @()

foreach ($package in $requiredPackages) {
    try {
        $packageName = $package -replace "-", "_"
        python -c "import $packageName" 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "   ✅ $package" -ForegroundColor Green
        } else {
            Write-Host "   ❌ $package (missing)" -ForegroundColor Red
            $missingPackages += $package
        }
    } catch {
        Write-Host "   ❌ $package (missing)" -ForegroundColor Red
        $missingPackages += $package
    }
}

if ($missingPackages.Count -gt 0) {
    Write-Host ""
    Write-Host "⚠️  Missing packages: $($missingPackages -join ', ')" -ForegroundColor Yellow
    Write-Host "   Please install them manually: pip install $($missingPackages -join ' ')" -ForegroundColor Yellow
} else {
    Write-Host ""
    Write-Host "✅ All required packages are installed!" -ForegroundColor Green
}

# Final instructions
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "✅ Setup completed successfully!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "📝 Next steps:" -ForegroundColor Yellow
Write-Host "   1. Edit .env file and add your OpenAI API key" -ForegroundColor White
Write-Host "   2. Run the application:" -ForegroundColor White
Write-Host "      streamlit run app.py" -ForegroundColor Cyan
Write-Host ""
Write-Host "🐳 Or use Docker:" -ForegroundColor Yellow
Write-Host "   docker build -t rag-chat-assistant ." -ForegroundColor Cyan
Write-Host "   docker run -p 8501:8501 -v `${PWD}/data:/app/data rag-chat-assistant" -ForegroundColor Cyan
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
