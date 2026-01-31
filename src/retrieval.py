# =============================================================================
# HYBRID RETRIEVAL SERVICE - Fetches Relevant Documents for the AI
# =============================================================================
#
# This module handles the "R" in RAG (Retrieval-Augmented Generation).
# When a user asks a question, this service finds the most relevant
# document chunks from the Chroma vector database.
#
# Architecture Flow:
#     User Question --> Retrieval Service --> Vector Store --> Relevant Chunks
#
# Hybrid Approach:
#     1. PRIMARY: MCP Server tools (if available) - structured tool-based access
#     2. FALLBACK: Direct Chroma vector search - semantic similarity search
#
# Why Hybrid?
#     - MCP provides standardized tool interface for complex operations
#     - Direct vector search is fast and reliable for simple queries
#     - Fallback ensures the system works even if MCP fails
#
# Key Concepts:
#     - Semantic Search: Finds documents by meaning, not just keywords
#     - Relevance Scoring: Converts distance to 0-1 similarity score
#     - Threshold Filtering: Only returns chunks above quality threshold
#
# =============================================================================

"""Hybrid retrieval system (MCP Server + Vector Store)."""

from typing import List, Dict, Any, Optional

from src.vector_store import VectorStore
from src.config import settings
from src.utils import get_logger

logger = get_logger(__name__)


class RetrievalService:
    """
    Hybrid retrieval service using MCP Server (primary) and vector store (fallback).
    
    This is the central service for finding relevant documents. It:
    1. Takes a user's question
    2. Converts it to a vector (embedding)
    3. Finds similar document chunks in the database
    4. Filters results by relevance quality
    5. Formats results for the AI to read
    
    The service uses a "hybrid" approach - it tries the MCP Server first
    (for standardized tool-based access), and falls back to direct vector
    search if MCP is unavailable.
    
    Example Usage:
        service = RetrievalService(vector_store)
        results = service.retrieve("What is their work experience?")
        context = service.get_context_text(results)
    """

    def __init__(self, vector_store: VectorStore, mcp_client=None):
        """
        Initialize the retrieval service.

        Args:
            vector_store: The Chroma vector store instance that holds all
                         document embeddings. This is where we search.
            mcp_client: Optional MCP client for tool-based retrieval.
                       If None, we use direct vector search.
        """
        self.vector_store = vector_store
        self.mcp_client = mcp_client
        
        # How many chunks to retrieve per query
        # 5 is a good balance - enough context without overwhelming the LLM
        self.top_k = settings.top_k_chunks
        
        # Minimum relevance score (0-1) to include a result
        # 0.3 is relatively permissive - we lowered it from 0.7 because
        # the stricter threshold was filtering out too many valid results
        self.relevance_threshold = settings.relevance_threshold

    def retrieve(
        self,
        query: str,
        use_mcp: bool = True,
        n_results: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant document chunks for a query.
        
        This is the main method you call to find relevant documents.
        It uses a hybrid approach:
        1. Try MCP Server first (if available and enabled)
        2. Fall back to direct vector store search
        3. Filter results by relevance threshold
        
        Args:
            query: The user's question or search text
            use_mcp: Whether to try MCP Server first (default: True)
            n_results: How many results to fetch (default: top_k from settings)
        
        Returns:
            List of chunk dictionaries, each containing:
            - id: Unique chunk identifier
            - text: The actual text content
            - metadata: Source file, chunk index, etc.
            - distance: How far from the query (lower = more similar)
            - relevance_score: 0-1 similarity (higher = more similar)
        
        Example:
            results = service.retrieve("What is their experience?")
            for r in results:
                print(f"{r['metadata']['source_file']}: {r['text'][:100]}...")
        """
        n_results = n_results or self.top_k

        # ----- Try MCP Server first (primary retrieval method) -----
        if use_mcp and self.mcp_client:
            try:
                results = self._retrieve_via_mcp(query, n_results)
                if results:
                    logger.info(f"Retrieved {len(results)} results via MCP Server")
                    return self._filter_by_relevance(results)
            except Exception as e:
                # MCP failed - log the error and try the fallback
                logger.warning(f"MCP retrieval failed: {e}, falling back to vector store")

        # ----- Fallback: Direct vector store search -----
        # This is the reliable fallback that always works
        logger.info("Using direct vector store search")
        results = self.vector_store.search(query=query, n_results=n_results)
        return self._filter_by_relevance(results)

    def _retrieve_via_mcp(self, query: str, n_results: int) -> List[Dict[str, Any]]:
        """
        Retrieve documents via MCP Server tools.
        
        MCP (Model Context Protocol) provides a standardized way for AI
        to call tools. This method would use the MCP client to call
        the search_documents tool on the MCP server.
        
        NOTE: This is currently a placeholder implementation.
        The actual MCP tool calling is TODO.
        
        Args:
            query: Search query text
            n_results: Number of results to return
        
        Returns:
            List of search results from MCP Server
        """
        # Future implementation would look like:
        # results = await self.mcp_client.call_tool(
        #     "search_documents", 
        #     {"query": query, "n_results": n_results}
        # )
        if self.mcp_client:
            # TODO: Implement MCP tool calling
            pass

        # For now, fall back to vector store
        return self.vector_store.search(query=query, n_results=n_results)

    def _filter_by_relevance(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter search results by relevance threshold.
        
        This is a quality filter that removes low-quality matches.
        It converts Chroma's distance metric to a 0-1 relevance score
        and only keeps results above the threshold.
        
        WHY THIS MATTERS:
        Without filtering, the AI might get irrelevant chunks that
        confuse it or lead to incorrect answers. The threshold ensures
        only high-quality matches are used.
        
        DISTANCE TO RELEVANCE CONVERSION:
        Chroma uses L2 (Euclidean) distance where:
        - 0 = identical (perfect match)
        - Higher = less similar
        
        We convert this to a 0-1 relevance score using:
            relevance = 1.0 / (1.0 + distance)
        
        This gives:
        - distance=0 → relevance=1.0 (perfect)
        - distance=1 → relevance=0.5
        - distance=9 → relevance=0.1
        
        Args:
            results: Raw search results from vector store
        
        Returns:
            Filtered results with relevance_score added
        """
        if not results:
            return []

        filtered = []
        for result in results:
            # Get the distance from Chroma (lower = more similar)
            distance = result.get("distance", 0.0)
            
            # Convert distance to 0-1 relevance score
            # Using 1/(1+d) formula for smooth decay
            relevance_score = 1.0 / (1.0 + distance)
            
            # Add the score to the result for transparency
            result["relevance_score"] = relevance_score

            # ----- Apply threshold filter -----
            if self.relevance_threshold <= 0:
                # Threshold disabled - keep all results
                filtered.append(result)
            elif relevance_score >= self.relevance_threshold:
                # Result passes quality check
                filtered.append(result)
            else:
                # Result filtered out - log for debugging
                logger.debug(
                    f"Filtered out result with relevance {relevance_score:.3f} "
                    f"(threshold: {self.relevance_threshold})"
                )

        # ----- Fallback: Don't return empty results -----
        # If the threshold filtered out EVERYTHING, that's probably too strict.
        # In this case, return the original results to ensure the AI has
        # something to work with.
        if not filtered and results:
            logger.warning(
                f"All {len(results)} results filtered out by threshold {self.relevance_threshold}, "
                "returning top results without filtering"
            )
            # Add relevance scores to original results
            for result in results:
                distance = result.get("distance", 0.0)
                result["relevance_score"] = 1.0 / (1.0 + distance)
            return results

        logger.info(f"Returning {len(filtered)} results after relevance filtering")
        return filtered

    def get_context_text(self, results: List[Dict[str, Any]]) -> str:
        """
        Format retrieved results into a context string for the LLM.
        
        This takes the raw search results and formats them into a
        human-readable string that can be injected into the prompt.
        Each chunk includes its source file and chunk number for
        citation purposes.
        
        Example Output:
            [Source: resume.pdf, Chunk 0]
            John Doe has 10 years of experience...
            
            [Source: resume.pdf, Chunk 2]
            Education: BS Computer Science...
        
        Args:
            results: List of retrieved chunk dictionaries
        
        Returns:
            Formatted context string, or "No relevant context found."
        """
        if not results:
            return "No relevant context found."

        context_parts = []
        for i, result in enumerate(results, 1):
            # Extract metadata for citation
            source = result.get("metadata", {}).get("source_file", "Unknown")
            chunk_idx = result.get("metadata", {}).get("chunk_index", "N/A")
            text = result.get("text", "")

            # Format with source citation
            # The AI will use this to cite its sources in the answer
            context_parts.append(
                f"[Source: {source}, Chunk {chunk_idx}]\n{text}\n"
            )

        return "\n".join(context_parts)
