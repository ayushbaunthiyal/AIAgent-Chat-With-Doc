"""MCP Server tools for document operations."""

from typing import Any, Dict, List
import json

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
except ImportError:
    # Fallback: Use alternative MCP implementation if package structure differs
    try:
        from mcp import Server, stdio_server, Tool, TextContent
    except ImportError:
        # Create minimal stubs for development
        class Server:
            pass
        class Tool:
            pass
        class TextContent:
            pass
        def stdio_server():
            pass

from src.vector_store import VectorStore
from src.config import settings
from src.utils import get_logger

logger = get_logger(__name__)

# Global vector store instance (will be initialized)
vector_store: VectorStore | None = None


def initialize_vector_store() -> None:
    """Initialize the vector store for MCP tools."""
    global vector_store
    if vector_store is None:
        vector_store = VectorStore()
        logger.info("Vector store initialized for MCP Server")


def get_document_context_tool() -> Tool:
    """Get document context tool definition."""
    return Tool(
        name="get_document_context",
        description=(
            "Retrieve relevant document chunks based on a query. "
            "Searches the Chroma vector store for semantically similar content."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant document chunks",
                },
                "n_results": {
                    "type": "integer",
                    "description": "Number of results to return (default: 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    )


def search_documents_tool() -> Tool:
    """Get search documents tool definition."""
    return Tool(
        name="search_documents",
        description=(
            "Search documents using semantic similarity. "
            "Returns the most relevant chunks from the document collection."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                },
                "n_results": {
                    "type": "integer",
                    "description": "Number of results to return (default: 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    )


def list_documents_tool() -> Tool:
    """Get list documents tool definition."""
    return Tool(
        name="list_documents",
        description="List all documents available in the collection.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    )


async def handle_get_document_context(arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle get_document_context tool call."""
    initialize_vector_store()

    query = arguments.get("query", "")
    n_results = arguments.get("n_results", settings.top_k_chunks)

    if not query:
        return [TextContent(type="text", text="Error: Query is required")]

    try:
        results = vector_store.search(query=query, n_results=n_results)

        if not results:
            return [TextContent(type="text", text="No relevant documents found.")]

        # Format results
        formatted_results = []
        for i, result in enumerate(results, 1):
            text = f"Result {i}:\n"
            text += f"Source: {result['metadata'].get('source_file', 'Unknown')}\n"
            text += f"Chunk Index: {result['metadata'].get('chunk_index', 'N/A')}\n"
            text += f"Content: {result['text']}\n"
            if result.get("distance"):
                text += f"Relevance Score: {1 - result['distance']:.3f}\n"
            formatted_results.append(TextContent(type="text", text=text))

        return formatted_results
    except Exception as e:
        logger.error(f"Error in get_document_context: {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def handle_search_documents(arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle search_documents tool call."""
    # This is essentially the same as get_document_context
    return await handle_get_document_context(arguments)


async def handle_list_documents(arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle list_documents tool call."""
    initialize_vector_store()

    try:
        documents = vector_store.list_documents()

        if not documents:
            return [TextContent(type="text", text="No documents in the collection.")]

        # Format document list
        text = f"Found {len(documents)} document(s):\n\n"
        for i, doc in enumerate(documents, 1):
            text += f"{i}. Document ID: {doc['document_id']}\n"
            text += f"   Source File: {doc.get('source_file', 'Unknown')}\n"
            text += f"   Timestamp: {doc.get('timestamp', 'N/A')}\n\n"

        return [TextContent(type="text", text=text)]
    except Exception as e:
        logger.error(f"Error in list_documents: {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]


# Tool handlers mapping
TOOL_HANDLERS = {
    "get_document_context": handle_get_document_context,
    "search_documents": handle_search_documents,
    "list_documents": handle_list_documents,
}
