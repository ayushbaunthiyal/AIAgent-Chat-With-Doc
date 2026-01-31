"""Hybrid retrieval system (MCP Server + Vector Store)."""

from typing import List, Dict, Any, Optional

from src.vector_store import VectorStore
from src.config import settings
from src.utils import get_logger

logger = get_logger(__name__)


class RetrievalService:
    """Hybrid retrieval service using MCP Server (primary) and vector store (fallback)."""

    def __init__(self, vector_store: VectorStore, mcp_client=None):
        """
        Initialize retrieval service.

        Args:
            vector_store: Chroma vector store instance
            mcp_client: Optional MCP client (for primary retrieval)
        """
        self.vector_store = vector_store
        self.mcp_client = mcp_client
        self.top_k = settings.top_k_chunks
        self.relevance_threshold = settings.relevance_threshold

    def retrieve(
        self,
        query: str,
        use_mcp: bool = True,
        n_results: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks using hybrid approach.

        Args:
            query: Search query
            use_mcp: Whether to use MCP Server (primary) or direct vector search
            n_results: Number of results (defaults to top_k)

        Returns:
            List of retrieved chunks with metadata
        """
        n_results = n_results or self.top_k

        # Try MCP Server first (if available and enabled)
        if use_mcp and self.mcp_client:
            try:
                results = self._retrieve_via_mcp(query, n_results)
                if results:
                    logger.info(f"Retrieved {len(results)} results via MCP Server")
                    return self._filter_by_relevance(results)
            except Exception as e:
                logger.warning(f"MCP retrieval failed: {e}, falling back to vector store")

        # Fallback to direct vector store search
        logger.info("Using direct vector store search")
        results = self.vector_store.search(query=query, n_results=n_results)
        return self._filter_by_relevance(results)

    def _retrieve_via_mcp(self, query: str, n_results: int) -> List[Dict[str, Any]]:
        """
        Retrieve via MCP Server.

        Args:
            query: Search query
            n_results: Number of results

        Returns:
            List of results from MCP Server
        """
        # This will be implemented when MCP client is fully integrated
        # For now, fall back to vector store
        if self.mcp_client:
            # TODO: Implement MCP tool calling
            # results = await self.mcp_client.call_tool("search_documents", {"query": query, "n_results": n_results})
            pass

        # Fallback to vector store
        return self.vector_store.search(query=query, n_results=n_results)

    def _filter_by_relevance(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter results by relevance threshold.

        Args:
            results: List of search results

        Returns:
            Filtered results above relevance threshold
        """
        if not results:
            return []

        filtered = []
        for result in results:
            # Calculate relevance score from distance
            # Chroma uses L2/cosine distance - smaller is better
            # Use exponential decay to convert distance to 0-1 similarity
            distance = result.get("distance", 0.0)
            relevance_score = 1.0 / (1.0 + distance)
            
            result["relevance_score"] = relevance_score

            # Only filter if threshold is set above 0
            if self.relevance_threshold <= 0:
                # No filtering, return all results
                filtered.append(result)
            elif relevance_score >= self.relevance_threshold:
                filtered.append(result)
            else:
                logger.debug(
                    f"Filtered out result with relevance {relevance_score:.3f} "
                    f"(threshold: {self.relevance_threshold})"
                )

        # If all filtered out, return original results (don't leave empty)
        if not filtered and results:
            logger.warning(
                f"All {len(results)} results filtered out by threshold {self.relevance_threshold}, "
                "returning top results without filtering"
            )
            for result in results:
                distance = result.get("distance", 0.0)
                result["relevance_score"] = 1.0 / (1.0 + distance)
            return results

        logger.info(f"Returning {len(filtered)} results after relevance filtering")
        return filtered

    def get_context_text(self, results: List[Dict[str, Any]]) -> str:
        """
        Format retrieved results into context text.

        Args:
            results: List of retrieved chunks

        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant context found."

        context_parts = []
        for i, result in enumerate(results, 1):
            source = result.get("metadata", {}).get("source_file", "Unknown")
            chunk_idx = result.get("metadata", {}).get("chunk_index", "N/A")
            text = result.get("text", "")

            context_parts.append(
                f"[Source: {source}, Chunk {chunk_idx}]\n{text}\n"
            )

        return "\n".join(context_parts)
