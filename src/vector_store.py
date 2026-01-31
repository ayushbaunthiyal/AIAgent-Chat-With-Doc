# =============================================================================
# CHROMA VECTOR STORE - The Document Database
# =============================================================================
#
# This module manages the Chroma vector database, which stores all document
# chunks and their embeddings. It's the "memory" of the RAG system.
#
# What is a Vector Database?
#     Unlike traditional databases that search by keywords, vector databases
#     search by meaning. Each document chunk is converted to a vector (a list
#     of numbers) that captures its semantic meaning. Similar documents have
#     similar vectors.
#
# Architecture:
#     Documents --> Chunks --> Embeddings --> Chroma DB --> Semantic Search
#
# Why Chroma?
#     - Free and open source (zero infrastructure cost)
#     - Persistent storage (data survives restarts)
#     - Easy setup (no external servers needed)
#     - Built-in embedding support with OpenAI
#
# Storage Location:
#     ./data/chroma_db/ contains:
#     - chroma.sqlite3: Metadata and chunk text
#     - HNSW index files: Fast similarity search index
#
# =============================================================================

"""Chroma vector store wrapper for persistent storage."""

from typing import List, Dict, Any, Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions

from src.config import settings
from src.utils import get_logger

logger = get_logger(__name__)


class VectorStore:
    """
    Chroma vector store manager with persistent storage.
    
    This class wraps the Chroma database and provides a clean interface for:
    - Adding document chunks (with automatic embedding)
    - Searching for similar documents
    - Managing the document collection
    
    The vector store uses OpenAI's text-embedding-3-small model to convert
    text into 1536-dimensional vectors. These vectors are stored in Chroma
    and indexed for fast similarity search.
    
    Example Usage:
        store = VectorStore()
        store.add_chunks(chunks)  # chunks = [{"id": "...", "text": "...", "metadata": {...}}]
        results = store.search("What is their experience?")
    """

    def __init__(self):
        """
        Initialize the Chroma vector store.
        
        This sets up:
        1. Persistent storage in ./data/chroma_db/
        2. OpenAI embedding function for converting text to vectors
        3. A collection named "documents" for storing chunks
        
        The database is created automatically if it doesn't exist.
        """
        self.db_path = settings.chroma_db_path_resolved
        self.collection_name = settings.chroma_collection_name

        # ----- Initialize Chroma client with persistent storage -----
        # PersistentClient saves data to disk, so it survives restarts
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=ChromaSettings(
                anonymized_telemetry=False,  # Disable usage tracking
                allow_reset=True,            # Allow clearing the database
            ),
        )

        # ----- Set up OpenAI embeddings -----
        # This function converts text to vectors using OpenAI's API
        # Model: text-embedding-3-small produces 1536-dimensional vectors
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=settings.openai_api_key,
            model_name=settings.openai_embedding_model,
        )

        # ----- Get or create the document collection -----
        # A collection is like a table in a traditional database
        # We use a single collection for all documents
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"description": "Document chunks with embeddings"},
        )

        logger.info(
            f"Initialized Chroma vector store at {self.db_path} "
            f"with collection '{self.collection_name}'"
        )

    def add_chunks(
        self, chunks: List[Dict[str, Any]], embeddings: Optional[List[List[float]]] = None
    ) -> None:
        """
        Add document chunks to the vector store.
        
        Each chunk is stored with its text, metadata, and embedding.
        If embeddings are not provided, they are automatically generated
        using the OpenAI embedding function.
        
        Args:
            chunks: List of chunk dictionaries, each containing:
                   - id: Unique identifier (e.g., "doc_abc123_chunk_0")
                   - text: The actual text content
                   - metadata: Source file, chunk index, timestamp, etc.
            embeddings: Optional pre-computed embeddings. If None, Chroma
                       will generate them automatically using OpenAI.
        
        Example:
            chunks = [
                {
                    "id": "doc_abc123_chunk_0",
                    "text": "John has 10 years of experience...",
                    "metadata": {"source_file": "resume.pdf", "chunk_index": 0}
                }
            ]
            store.add_chunks(chunks)
        """
        if not chunks:
            return

        try:
            # Extract components from chunk dictionaries
            ids = [chunk["id"] for chunk in chunks]
            texts = [chunk["text"] for chunk in chunks]
            metadatas = [chunk["metadata"] for chunk in chunks]

            # Add to Chroma collection
            # If embeddings are provided, use them; otherwise Chroma generates them
            if embeddings:
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                )
            else:
                # Let Chroma generate embeddings using the OpenAI function
                # This calls the OpenAI API internally
                self.collection.add(
                    ids=ids,
                    documents=texts,
                    metadatas=metadatas,
                )

            logger.info(f"Added {len(chunks)} chunks to vector store")
        except Exception as e:
            logger.error(f"Error adding chunks to vector store: {e}")
            raise

    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar document chunks using semantic similarity.
        
        HOW IT WORKS:
        1. Your query is converted to a vector (embedding)
        2. Chroma finds the n_results closest vectors in the database
        3. Results are returned sorted by similarity (closest first)
        
        The search uses cosine/L2 distance - chunks with similar meaning
        have similar vectors and thus lower distance scores.
        
        Args:
            query: The search text (e.g., "What is their work experience?")
            n_results: How many results to return (default: 5)
            filter_metadata: Optional filters like {"source_file": "resume.pdf"}
        
        Returns:
            List of result dictionaries, each containing:
            - id: Chunk identifier
            - text: The chunk text content
            - metadata: Source file, chunk index, etc.
            - distance: Similarity distance (lower = more similar)
        
        Example:
            results = store.search("What is their experience?", n_results=3)
            for r in results:
                print(f"Distance: {r['distance']:.3f}")
                print(f"Text: {r['text'][:100]}...")
        """
        try:
            # Query the Chroma collection
            # This automatically embeds the query and finds similar chunks
            results = self.collection.query(
                query_texts=[query],     # Chroma embeds this automatically
                n_results=n_results,     # How many results to return
                where=filter_metadata,   # Optional metadata filter
            )

            # ----- Format results into a cleaner structure -----
            # Chroma returns nested lists because it supports batch queries
            # We only query one thing at a time, so we extract from index [0]
            formatted_results = []
            if results["ids"] and len(results["ids"][0]) > 0:
                for i in range(len(results["ids"][0])):
                    formatted_results.append(
                        {
                            "id": results["ids"][0][i],
                            "text": results["documents"][0][i],
                            "metadata": results["metadatas"][0][i],
                            "distance": results["distances"][0][i] if results["distances"] else None,
                        }
                    )

            logger.info(f"Found {len(formatted_results)} results for query")
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            raise

    def get_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve specific chunks by their IDs.
        
        Use this when you know exactly which chunks you want,
        rather than searching by similarity.
        
        Args:
            ids: List of chunk IDs to retrieve
        
        Returns:
            List of chunk dictionaries with id, text, and metadata
        """
        try:
            results = self.collection.get(ids=ids, include=["documents", "metadatas"])

            chunks = []
            if results["ids"]:
                for i in range(len(results["ids"])):
                    chunks.append(
                        {
                            "id": results["ids"][i],
                            "text": results["documents"][i],
                            "metadata": results["metadatas"][i],
                        }
                    )

            return chunks
        except Exception as e:
            logger.error(f"Error retrieving chunks by IDs: {e}")
            raise

    def delete_by_document_id(self, document_id: str) -> None:
        """
        Delete all chunks belonging to a specific document.
        
        This is useful when you want to remove a document and all its
        chunks from the database (e.g., when re-processing a file).
        
        Args:
            document_id: The document ID to delete (e.g., "doc_abc123")
        """
        try:
            # Find all chunks with this document_id in their metadata
            results = self.collection.get(
                where={"document_id": document_id},
                include=["ids"],
            )

            if results["ids"]:
                # Delete all matching chunks
                self.collection.delete(ids=results["ids"])
                logger.info(f"Deleted {len(results['ids'])} chunks for document {document_id}")
        except Exception as e:
            logger.error(f"Error deleting document chunks: {e}")
            raise

    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all unique documents in the collection.
        
        This scans all chunks and extracts unique document IDs,
        useful for showing what documents have been uploaded.
        
        Returns:
            List of document info dictionaries with:
            - document_id: Unique document identifier
            - source_file: Original file name
            - timestamp: When it was added
        """
        try:
            # Get all chunks (we just need metadata, not the full text)
            all_items = self.collection.get(include=["metadatas"])

            # Extract unique documents by document_id
            documents = {}
            if all_items["metadatas"]:
                for metadata in all_items["metadatas"]:
                    doc_id = metadata.get("document_id")
                    if doc_id and doc_id not in documents:
                        documents[doc_id] = {
                            "document_id": doc_id,
                            "source_file": metadata.get("source_file"),
                            "timestamp": metadata.get("timestamp"),
                        }

            return list(documents.values())
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            raise

    def get_collection_count(self) -> int:
        """
        Get the total number of chunks in the collection.
        
        Returns:
            Number of stored chunks (not documents - one document = many chunks)
        """
        return self.collection.count()

    def clear_all(self) -> None:
        """
        Delete ALL chunks from the collection.
        
        WARNING: This removes all data! Use with caution.
        The collection itself remains, just empty.
        
        This is called when the user clicks "Clear Collection" in the UI.
        """
        try:
            # Get all chunk IDs
            # NOTE: We don't use include=["ids"] because Chroma returns IDs by default
            # Including it actually causes an error in newer Chroma versions
            all_items = self.collection.get()

            if all_items["ids"]:
                # Delete all chunks
                self.collection.delete(ids=all_items["ids"])
                logger.info(f"Cleared {len(all_items['ids'])} chunks from collection")
            else:
                logger.info("Collection is already empty")
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            raise
