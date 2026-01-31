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

# Set up logging
setup_logging()
logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="RAG Chat Assistant",
    page_icon="📚",
    layout="wide",
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "agent" not in st.session_state:
    st.session_state.agent = None

if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False

# Note: Chat history is managed in st.session_state.messages
# LangGraph agent processes each query independently (no checkpointer)


def initialize_components():
    """Initialize vector store and agent components."""
    if st.session_state.vector_store is None:
        with st.spinner("Initializing vector store..."):
            st.session_state.vector_store = VectorStore()
            logger.info("Vector store initialized")

    if st.session_state.agent is None:
        with st.spinner("Initializing agent..."):
            retrieval_service = RetrievalService(st.session_state.vector_store)
            st.session_state.agent = ReActAgent(retrieval_service)
            logger.info("Agent initialized")


def process_uploaded_file(uploaded_file) -> bool:
    """
    Process an uploaded file and add to vector store.

    Args:
        uploaded_file: Streamlit uploaded file object

    Returns:
        True if successful, False otherwise
    """
    try:
        # Validate file extension
        allowed_extensions = [".pdf", ".txt", ".md"]
        if not validate_file_extension(uploaded_file.name, allowed_extensions):
            st.error(
                f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )
            return False

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            # Process document
            processor = DocumentProcessor()
            chunks = processor.process_document(tmp_path)

            if not chunks:
                st.warning("No content extracted from the document.")
                return False

            # Add to vector store
            initialize_components()
            st.session_state.vector_store.add_chunks(chunks)

            st.success(
                f"✅ Document '{uploaded_file.name}' processed successfully! "
                f"Added {len(chunks)} chunks to the collection."
            )
            logger.info(f"Processed document: {uploaded_file.name} ({len(chunks)} chunks)")
            return True

        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        logger.error(f"Error processing file {uploaded_file.name}: {e}")
        return False


def main():
    """Main Streamlit application."""
    st.title("📚 RAG Chat Assistant")
    st.markdown(
        "Ask questions about your uploaded documents using AI-powered retrieval."
    )

    # Sidebar for document management
    with st.sidebar:
        st.header("📄 Document Management")

        # File upload
        uploaded_file = st.file_uploader(
            "Upload a document",
            type=["pdf", "txt", "md"],
            help="Upload PDF, TXT, or MD files",
        )

        if uploaded_file is not None:
            if st.button("Process Document", type="primary"):
                process_uploaded_file(uploaded_file)
                st.session_state.documents_loaded = True

        st.divider()

        # Document info
        if st.session_state.vector_store is not None:
            try:
                doc_count = st.session_state.vector_store.get_collection_count()
                st.metric("Total Chunks", doc_count)

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

        # Clear collection
        if st.button("🗑️ Clear Collection", type="secondary"):
            if st.session_state.vector_store is not None:
                try:
                    # Get count before clearing
                    count_before = st.session_state.vector_store.get_collection_count()

                    # Clear all chunks
                    st.session_state.vector_store.clear_all()

                    # Clear chat messages
                    st.session_state.messages = []

                    st.success(f"✅ Cleared {count_before} chunks from the collection!")
                    st.rerun()  # Refresh the page to update the UI
                except Exception as e:
                    st.error(f"Error clearing collection: {str(e)}")
                    logger.error(f"Error clearing collection: {e}")
            else:
                st.warning("No collection to clear.")

    # Initialize components
    initialize_components()

    # Main chat interface
    st.header("💬 Chat")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Sanitize input
        prompt = sanitize_text(prompt)

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Check if documents are loaded
                    if st.session_state.vector_store is None:
                        response = (
                            "Please upload and process a document first before asking questions."
                        )
                    else:
                        # Get response from agent (no checkpointer - history managed here)
                        response = st.session_state.agent.invoke(prompt)

                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    logger.info(f"Generated response for query: {prompt[:50]}...")

                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    logger.error(f"Error generating response: {e}")


if __name__ == "__main__":
    main()
