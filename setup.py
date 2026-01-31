"""Setup script for RAG Chat Assistant."""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def check_python_version():
    """Check if Python version is 3.10 or higher."""
    if sys.version_info < (3, 10):
        print("❌ Python 3.10 or higher is required.")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True


def check_uv_installed():
    """Check if UV is installed."""
    try:
        result = subprocess.run(
            ["uv", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"✅ UV is installed: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  UV is not installed.")
        return False


def install_uv():
    """Install UV package manager."""
    print("\n📦 Installing UV...")
    try:
        if sys.platform == "win32":
            # Windows installation
            subprocess.run(
                [
                    "powershell",
                    "-Command",
                    "irm https://astral.sh/uv/install.ps1 | iex",
                ],
                check=True,
            )
        else:
            # Unix-like installation
            subprocess.run(
                ["curl", "-LsSf", "https://astral.sh/uv/install.sh", "|", "sh"],
                shell=True,
                check=True,
            )
        print("✅ UV installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install UV: {e}")
        return False


def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    directories = [
        "data",
        "data/chroma_db",
        "tests",
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Created/verified: {directory}")


def create_env_file():
    """Create .env file from .env.example if it doesn't exist."""
    print("\n🔧 Setting up environment file...")
    env_file = Path(".env")
    env_example = Path(".env.example")

    if env_file.exists():
        print("   ℹ️  .env file already exists, skipping...")
        return

    if env_example.exists():
        # Copy .env.example to .env
        shutil.copy(env_example, env_file)
        print("   ✅ Created .env from .env.example")
        print("   ⚠️  Please edit .env and add your OpenAI API key!")
    else:
        # Create a basic .env file
        env_content = """# OpenAI API Configuration
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
"""
        env_file.write_text(env_content)
        print("   ✅ Created .env file")
        print("   ⚠️  Please edit .env and add your OpenAI API key!")


def install_dependencies(use_uv=True):
    """Install project dependencies."""
    print("\n📦 Installing dependencies...")
    try:
        if use_uv:
            print("   Using UV package manager...")
            subprocess.run(
                ["uv", "pip", "install", "-e", "."],
                check=True,
            )
        else:
            print("   Using pip...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", "."],
                check=True,
            )
        print("   ✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed to install dependencies: {e}")
        return False


def verify_installation():
    """Verify that key packages are installed."""
    print("\n🔍 Verifying installation...")
    required_packages = [
        "openai",
        "langchain",
        "langgraph",
        "chromadb",
        "streamlit",
        "pypdf",
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} (missing)")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        return False

    print("\n✅ All required packages are installed!")
    return True


def main():
    """Main setup function."""
    print("=" * 60)
    print("🚀 RAG Chat Assistant - Setup Script")
    print("=" * 60)

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Check/install UV
    use_uv = check_uv_installed()
    if not use_uv:
        response = input("\n❓ UV is not installed. Install it now? (y/n): ")
        if response.lower() == "y":
            if install_uv():
                use_uv = True
            else:
                print("⚠️  Continuing with pip instead...")
                use_uv = False
        else:
            print("⚠️  Continuing with pip instead...")
            use_uv = False

    # Create directories
    create_directories()

    # Create .env file
    create_env_file()

    # Install dependencies
    if not install_dependencies(use_uv=use_uv):
        print("\n❌ Setup failed during dependency installation.")
        sys.exit(1)

    # Verify installation
    if not verify_installation():
        print("\n⚠️  Some packages may be missing. Please check the installation.")
        sys.exit(1)

    # Final instructions
    print("\n" + "=" * 60)
    print("✅ Setup completed successfully!")
    print("=" * 60)
    print("\n📝 Next steps:")
    print("   1. Edit .env file and add your OpenAI API key")
    print("   2. Run the application:")
    print("      streamlit run app.py")
    print("\n🐳 Or use Docker:")
    print("   docker build -t rag-chat-assistant .")
    print("   docker run -p 8501:8501 -v $(pwd)/data:/app/data rag-chat-assistant")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
