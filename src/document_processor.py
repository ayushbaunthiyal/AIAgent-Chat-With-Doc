"""Document processing for PDF and text files."""

import hashlib
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

import pypdf
from src.config import settings
from src.utils import get_logger, validate_file_extension

logger = get_logger(__name__)


class DocumentProcessor:
    """Process documents (PDF, text) and chunk them."""

    def __init__(self):
        """Initialize document processor."""
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
        self.allowed_extensions = [".pdf", ".txt", ".md"]

    def load_document(self, file_path: str) -> str:
        """
        Load document content from file.

        Args:
            file_path: Path to the document file

        Returns:
            Document content as string

        Raises:
            ValueError: If file extension is not supported
            FileNotFoundError: If file doesn't exist
        """
        file_path_obj = Path(file_path)

        if not file_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not validate_file_extension(file_path, self.allowed_extensions):
            raise ValueError(
                f"Unsupported file type. Allowed: {', '.join(self.allowed_extensions)}"
            )

        extension = file_path_obj.suffix.lower()

        if extension == ".pdf":
            return self._load_pdf(file_path)
        elif extension in [".txt", ".md"]:
            return self._load_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {extension}")

    def _load_pdf(self, file_path: str) -> str:
        """Load content from PDF file."""
        try:
            text_content = []
            with open(file_path, "rb") as file:
                pdf_reader = pypdf.PdfReader(file)
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text.strip():
                        text_content.append(text)
            return "\n\n".join(text_content)
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            raise

    def _load_text(self, file_path: str) -> str:
        """Load content from text file."""
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {e}")
            raise

    def chunk_text(
        self, text: str, source_file: str, document_id: str | None = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk text into smaller pieces with overlap.

        Args:
            text: Text content to chunk
            source_file: Source file path
            document_id: Optional document ID (generated if not provided)

        Returns:
            List of chunk dictionaries with text, metadata, and IDs
        """
        if not text.strip():
            return []

        # Generate document ID if not provided
        if document_id is None:
            document_id = self._generate_document_id(source_file)

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size

            # Extract chunk
            chunk_text = text[start:end].strip()

            if not chunk_text:
                break

            # Generate chunk ID
            chunk_id = self._generate_chunk_id(document_id, chunk_index)

            # Create chunk metadata
            chunk_metadata = {
                "document_id": document_id,
                "chunk_index": chunk_index,
                "source_file": source_file,
                "chunk_size": len(chunk_text),
                "timestamp": datetime.utcnow().isoformat(),
            }

            chunks.append(
                {
                    "id": chunk_id,
                    "text": chunk_text,
                    "metadata": chunk_metadata,
                }
            )

            # Move start position (with overlap)
            start = end - self.chunk_overlap
            chunk_index += 1

        logger.info(
            f"Chunked document {source_file} into {len(chunks)} chunks "
            f"(size: {self.chunk_size}, overlap: {self.chunk_overlap})"
        )

        return chunks

    def process_document(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Process a document: load, chunk, and return chunks.

        Args:
            file_path: Path to the document file

        Returns:
            List of chunk dictionaries
        """
        logger.info(f"Processing document: {file_path}")

        # Load document
        text_content = self.load_document(file_path)

        # Chunk text
        chunks = self.chunk_text(text_content, file_path)

        return chunks

    def _generate_document_id(self, source_file: str) -> str:
        """Generate a unique document ID from file path."""
        file_path_str = str(Path(source_file).resolve())
        hash_obj = hashlib.md5(file_path_str.encode())
        return f"doc_{hash_obj.hexdigest()[:12]}"

    def _generate_chunk_id(self, document_id: str, chunk_index: int) -> str:
        """Generate a unique chunk ID."""
        return f"{document_id}_chunk_{chunk_index}"
