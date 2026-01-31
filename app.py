# =============================================================================
# RAG CHAT ASSISTANT - Main Streamlit Application
# =============================================================================
#
# This is the entry point of the application. It creates the web interface
# that users interact with to upload documents and ask questions.
#
# How It Works:
#     1. User uploads a PDF/TXT/MD file
#     2. File is chunked and stored in Chroma vector database
#     3. User asks questions in the chat interface
#     4. The AI retrieves relevant chunks and generates answers
#
# Key Components:
#     - Streamlit: Web UI framework (handles all the visual stuff)
#     - Session State: Stores data between user interactions
#     - Vector Store: Chroma database for document storage
#     - Agent: LangGraph ReAct agent for question answering
#
# Session State Variables:
#     - messages: Chat history displayed in the UI
#     - vector_store: Chroma database instance
#     - agent: The ReAct agent instance
#     - documents_loaded: Flag indicating if any documents exist
#
# Why Session State?
#     Streamlit re-runs the entire script on every user interaction.
#     Session state persists data between these re-runs. Without it,
#     we'd lose the chat history and have to reinitialize everything.
#
# =============================================================================

"""Streamlit web interface for RAG Chat Assistant."""

import streamlit as st
from pathlib import Path
import tempfile
import os

from src.config import settings
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.retrieval import RetrievalService
from src.agent import ReActAgent
from src.utils import setup_logging, get_logger, validate_file_extension, sanitize_text

# Set up logging for debugging and monitoring
setup_logging()
logger = get_logger(__name__)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
# This must be the first Streamlit command in the script

st.set_page_config(
    page_title="RAG Chat Assistant",
    page_icon="📚",
    layout="wide",  # Use full screen width
)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
# Session state persists data between Streamlit re-runs.
# We initialize these variables only if they don't already exist.

# Chat message history - displayed in the chat interface
# Format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chroma vector store instance - holds all document embeddings
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# LangGraph ReAct agent - the AI that answers questions
if "agent" not in st.session_state:
    st.session_state.agent = None

# Flag to track if any documents have been loaded
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False

# NOTE: We don't use LangGraph's MemorySaver checkpointer.
# Chat history is managed here in session_state.messages instead.
# This prevents the context overflow issues we encountered earlier.


def initialize_components():
    """
    Initialize the vector store and agent if not already done.
    
    This is called lazily (only when needed) to avoid slow startup.
    The components are stored in session_state so they persist.
    
    Initialization Flow:
        1. Create VectorStore (connects to Chroma)
        2. Create RetrievalService (wraps vector store)
        3. Create ReActAgent (uses retrieval service)
    """
    # Initialize vector store (Chroma database connection)
    if st.session_state.vector_store is None:
        with st.spinner("Initializing vector store..."):
            st.session_state.vector_store = VectorStore()
            logger.info("Vector store initialized")

    # Initialize the ReAct agent
    if st.session_state.agent is None:
        with st.spinner("Initializing agent..."):
            # Create retrieval service that searches the vector store
            retrieval_service = RetrievalService(st.session_state.vector_store)
            # Create the agent with the retrieval service
            st.session_state.agent = ReActAgent(retrieval_service)
            logger.info("Agent initialized")


def process_uploaded_file(uploaded_file) -> bool:
    """
    Process an uploaded document and add it to the vector store.
    
    Processing Steps:
        1. Validate file extension (must be PDF, TXT, or MD)
        2. Save to temporary file (Streamlit uploads are in memory)
        3. Load and chunk the document (1000 chars, 200 overlap)
        4. Store chunks in Chroma (embeddings generated automatically)
        5. Clean up temporary file
    
    Args:
        uploaded_file: Streamlit UploadedFile object from file_uploader
    
    Returns:
        True if processing succeeded, False otherwise
    
    Error Handling:
        - Invalid file type: Shows error message
        - Empty document: Shows warning message
        - Processing errors: Logs and shows error message
    """
    try:
        # ----- Step 1: Validate file extension -----
        allowed_extensions = [".pdf", ".txt", ".md"]
        if not validate_file_extension(uploaded_file.name, allowed_extensions):
            st.error(
                f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )
            return False

        # ----- Step 2: Save to temporary file -----
        # Streamlit's uploaded file is in memory, but our processor needs a file path
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            # ----- Step 3: Process the document -----
            # This loads the file and splits it into chunks
            processor = DocumentProcessor()
            chunks = processor.process_document(tmp_path)

            if not chunks:
                st.warning("No content extracted from the document.")
                return False

            # ----- Step 4: Add chunks to vector store -----
            # Make sure components are initialized
            initialize_components()
            # Store chunks (embeddings generated automatically by Chroma)
            st.session_state.vector_store.add_chunks(chunks)

            # Show success message with chunk count
            st.success(
                f"✅ Document '{uploaded_file.name}' processed successfully! "
                f"Added {len(chunks)} chunks to the collection."
            )
            logger.info(f"Processed document: {uploaded_file.name} ({len(chunks)} chunks)")
            return True

        finally:
            # ----- Step 5: Clean up temporary file -----
            # Always delete the temp file, even if processing failed
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        logger.error(f"Error processing file {uploaded_file.name}: {e}")
        return False


def main():
    """
    Main application function - builds the Streamlit UI.
    
    UI Layout:
        ┌─────────────────────────────────────────────────────────┐
        │  📚 RAG Chat Assistant                                  │
        ├────────────────┬────────────────────────────────────────┤
        │   SIDEBAR      │              MAIN AREA                 │
        │                │                                        │
        │  📄 Document   │  💬 Chat                               │
        │  Management    │                                        │
        │                │  [Chat messages displayed here]        │
        │  [Upload]      │                                        │
        │  [Process]     │                                        │
        │                │                                        │
        │  Total Chunks  │  [Input box at bottom]                 │
        │  [List Docs]   │                                        │
        │  [Clear]       │                                        │
        └────────────────┴────────────────────────────────────────┘
    """
    # ----- Page Title -----
    st.title("📚 RAG Chat Assistant")
    st.markdown(
        "Ask questions about your uploaded documents using AI-powered retrieval."
    )

    # =========================================================================
    # SIDEBAR - Document Management
    # =========================================================================
    with st.sidebar:
        st.header("📄 Document Management")

        # ----- File Upload Widget -----
        uploaded_file = st.file_uploader(
            "Upload a document",
            type=["pdf", "txt", "md"],
            help="Upload PDF, TXT, or MD files",
        )

        # Process button appears when a file is selected
        if uploaded_file is not None:
            if st.button("Process Document", type="primary"):
                process_uploaded_file(uploaded_file)
                st.session_state.documents_loaded = True

        st.divider()

        # ----- Document Statistics -----
        if st.session_state.vector_store is not None:
            try:
                # Show total number of chunks in the database
                doc_count = st.session_state.vector_store.get_collection_count()
                st.metric("Total Chunks", doc_count)

                # List documents button
                if st.button("List Documents"):
                    documents = st.session_state.vector_store.list_documents()
                    if documents:
                        st.write("**Available Documents:**")
                        for doc in documents:
                            st.write(f"- {doc.get('source_file', 'Unknown')}")
                    else:
                        st.info("No documents in collection.")
            except Exception as e:
                st.error(f"Error getting document info: {e}")

        st.divider()

        # ----- Clear Collection Button -----
        # WARNING: This deletes all data!
        if st.button("🗑️ Clear Collection", type="secondary"):
            if st.session_state.vector_store is not None:
                try:
                    # Get count before clearing (for the success message)
                    count_before = st.session_state.vector_store.get_collection_count()

                    # Delete all chunks from Chroma
                    st.session_state.vector_store.clear_all()

                    # Also clear chat history since it's now irrelevant
                    st.session_state.messages = []

                    st.success(f"✅ Cleared {count_before} chunks from the collection!")
                    
                    # Rerun the app to refresh the UI
                    st.rerun()
                except Exception as e:
                    st.error(f"Error clearing collection: {str(e)}")
                    logger.error(f"Error clearing collection: {e}")
            else:
                st.warning("No collection to clear.")

    # =========================================================================
    # MAIN AREA - Chat Interface
    # =========================================================================
    
    # Initialize components (vector store and agent)
    initialize_components()

    st.header("💬 Chat")

    # ----- Display Chat History -----
    # Show all previous messages in the conversation
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ----- Chat Input -----
    # This creates an input box at the bottom of the chat
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Sanitize input to prevent prompt injection attacks
        prompt = sanitize_text(prompt)

        # Add user message to history and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # ----- Generate AI Response -----
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Check if we have a vector store with documents
                    if st.session_state.vector_store is None:
                        response = (
                            "Please upload and process a document first before asking questions."
                        )
                    else:
                        # Run the ReAct agent to get an answer
                        # The agent will:
                        # 1. Search for relevant document chunks
                        # 2. Generate an answer based on the context
                        response = st.session_state.agent.invoke(prompt)

                    # Display the response
                    st.markdown(response)
                    
                    # Add to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    logger.info(f"Generated response for query: {prompt[:50]}...")

                except Exception as e:
                    # Handle any errors gracefully
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    logger.error(f"Error generating response: {e}")


# =============================================================================
# ENTRY POINT
# =============================================================================
# This runs when you execute: streamlit run app.py

if __name__ == "__main__":
    main()
