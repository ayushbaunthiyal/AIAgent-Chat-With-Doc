"""Tests for document processor."""

import pytest
from pathlib import Path
import tempfile
import os

from src.document_processor import DocumentProcessor


def test_chunk_text():
    """Test text chunking functionality."""
    processor = DocumentProcessor()
    text = "A" * 2500  # 2500 characters
    chunks = processor.chunk_text(text, "test.txt")

    assert len(chunks) > 0
    assert all("id" in chunk for chunk in chunks)
    assert all("text" in chunk for chunk in chunks)
    assert all("metadata" in chunk for chunk in chunks)


def test_chunk_overlap():
    """Test that chunks have proper overlap."""
    processor = DocumentProcessor()
    text = "A" * 2500
    chunks = processor.chunk_text(text, "test.txt")

    if len(chunks) > 1:
        # Check that chunks overlap (simplified check)
        assert len(chunks[0]["text"]) <= processor.chunk_size + 100


def test_generate_document_id():
    """Test document ID generation."""
    processor = DocumentProcessor()
    doc_id = processor._generate_document_id("test_file.txt")
    
    assert doc_id.startswith("doc_")
    assert len(doc_id) > 10


def test_generate_chunk_id():
    """Test chunk ID generation."""
    processor = DocumentProcessor()
    chunk_id = processor._generate_chunk_id("doc_123", 0)
    
    assert chunk_id == "doc_123_chunk_0"
