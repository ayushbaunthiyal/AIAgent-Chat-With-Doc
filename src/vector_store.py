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
    """Chroma vector store manager with persistent storage."""

    def __init__(self):
        """Initialize Chroma vector store."""
        self.db_path = settings.chroma_db_path_resolved
        self.collection_name = settings.chroma_collection_name

        # Initialize Chroma client with persistent storage
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        # Use OpenAI embedding function
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=settings.openai_api_key,
            model_name=settings.openai_embedding_model,
        )

        # Get or create collection
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
        Add chunks to the vector store.

        Args:
            chunks: List of chunk dictionaries with id, text, and metadata
            embeddings: Optional pre-computed embeddings (if None, will be generated)
        """
        if not chunks:
            return

        try:
            ids = [chunk["id"] for chunk in chunks]
            texts = [chunk["text"] for chunk in chunks]
            metadatas = [chunk["metadata"] for chunk in chunks]

            # Add to collection (embeddings will be generated automatically if not provided)
            if embeddings:
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                )
            else:
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
        Search for similar chunks using semantic similarity.

        Args:
            query: Search query text
            n_results: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            List of result dictionaries with text, metadata, and distance
        """
        try:
            # Query the collection
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=filter_metadata,
            )

            # Format results
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
        Retrieve chunks by their IDs.

        Args:
            ids: List of chunk IDs

        Returns:
            List of chunk dictionaries
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
        Delete all chunks for a specific document.

        Args:
            document_id: Document ID to delete
        """
        try:
            # Get all chunks for this document
            results = self.collection.get(
                where={"document_id": document_id},
                include=["ids"],
            )

            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                logger.info(f"Deleted {len(results['ids'])} chunks for document {document_id}")
        except Exception as e:
            logger.error(f"Error deleting document chunks: {e}")
            raise

    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all unique documents in the collection.

        Returns:
            List of document metadata dictionaries
        """
        try:
            # Get all items to extract unique documents
            all_items = self.collection.get(include=["metadatas"])

            # Extract unique documents
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
        """Get total number of chunks in the collection."""
        return self.collection.count()

    def clear_all(self) -> None:
        """
        Delete all chunks from the collection.
        """
        try:
            # Get all chunk IDs (ids are returned by default, no need to include)
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
