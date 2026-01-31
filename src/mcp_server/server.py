"""MCP Server implementation for document operations."""

import asyncio
import sys
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
except ImportError:
    # Fallback for development
    class Server:
        def __init__(self, name):
            self.name = name
        def list_tools(self):
            pass
        def call_tool(self):
            pass
        def run(self, *args, **kwargs):
            pass
        def create_initialization_options(self):
            return {}
    async def stdio_server():
        return (None, None)

from src.mcp_server.tools import (
    get_document_context_tool,
    search_documents_tool,
    list_documents_tool,
    TOOL_HANDLERS,
)
from src.utils import get_logger, setup_logging

# Set up logging
setup_logging()
logger = get_logger(__name__)

# Create MCP server
app = Server("rag-document-server")


@app.list_tools()
async def list_tools() -> list:
    """List available tools."""
    return [
        get_document_context_tool(),
        search_documents_tool(),
        list_documents_tool(),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list:
    """Handle tool calls."""
    logger.info(f"Tool called: {name} with arguments: {arguments}")

    if name not in TOOL_HANDLERS:
        raise ValueError(f"Unknown tool: {name}")

    handler = TOOL_HANDLERS[name]
    return await handler(arguments)


async def main():
    """Run the MCP server."""
    logger.info("Starting MCP Server...")
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
