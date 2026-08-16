"""
Document Processing and Chunking for ISMS RAG Platform
======================================================

This module provides comprehensive document processing capabilities for the ISMS RAG platform.
It handles loading, chunking, and metadata extraction for various document formats including
PDF, DOCX, and TXT files.

The DocumentProcessor class uses LangChain's RecursiveCharacterTextSplitter to intelligently
split documents into chunks while preserving semantic meaning and maintaining metadata.

Features:
    - Multiple format support (PDF, DOCX, TXT)
    - Intelligent text chunking with overlap
    - Comprehensive metadata tracking
    - Progress indicators for batch processing
    - Robust error handling
    - Framework and policy categorization

Usage:
    from src.config import Config
    from src.document_processor import DocumentProcessor
    
    processor = DocumentProcessor(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    )
    
    # Process single document
    chunks = processor.process_document("data/frameworks/iso27001/ISO27001.pdf")
    
    # Process entire directory
    all_chunks = processor.process_directory("data/frameworks/iso27001")

Date: 2026
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from tqdm import tqdm

from src.config import Config

# Configure logger
logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    """
    Metadata container for processed documents.
    
    Attributes:
        source: Full path to source document
        filename: Name of the source file
        file_type: Document type (pdf, docx, txt)
        framework: Framework category (iso27001, nist_csf, cis_controls, policy)
        chunk_id: Unique identifier for the chunk
        total_chunks: Total number of chunks in the document
        page_number: Page number (for PDFs)
        char_count: Number of characters in chunk
        metadata_extra: Additional custom metadata
    """
    source: str
    filename: str
    file_type: str
    framework: Optional[str] = None
    chunk_id: int = 0
    total_chunks: int = 0
    page_number: Optional[int] = None
    char_count: int = 0
    metadata_extra: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert metadata to dictionary format."""
        return {
            "source": self.source,
            "filename": self.filename,
            "file_type": self.file_type,
            "framework": self.framework,
            "chunk_id": self.chunk_id,
            "total_chunks": self.total_chunks,
            "page_number": self.page_number,
            "char_count": self.char_count,
            **self.metadata_extra,
        }


@dataclass
class ProcessedChunk:
    """
    Container for a processed document chunk.
    
    Attributes:
        content: Text content of the chunk
        metadata: Associated metadata
    """
    content: str
    metadata: DocumentMetadata
    
    def to_dict(self) -> Dict:
        """Convert chunk to dictionary format."""
        return {
            "content": self.content,
            "metadata": self.metadata.to_dict(),
        }


class DocumentProcessor:
    """
    Process and chunk documents for RAG applications.
    
    This class handles the complete document processing pipeline including:
    - Loading documents from various formats
    - Intelligent text chunking
    - Metadata extraction and tracking
    - Batch processing of directories
    
    Attributes:
        chunk_size: Target size for text chunks (in characters)
        chunk_overlap: Number of overlapping characters between chunks
        text_splitter: LangChain text splitter instance
        supported_formats: Set of supported file extensions
    """
    
    # Supported file formats
    SUPPORTED_FORMATS = {".pdf", ".docx", ".txt"}
    
    # Framework detection patterns
    FRAMEWORK_PATTERNS = {
        "iso27001": ["iso27001", "iso 27001", "iso_27001"],
        "nist_csf": ["nist", "csf", "cybersecurity framework"],
        "cis_controls": ["cis", "controls", "cis controls"],
    }
    
    def __init__(
        self,
        chunk_size: int = Config.CHUNK_SIZE,
        chunk_overlap: int = Config.CHUNK_OVERLAP,
    ):
        """
        Initialize DocumentProcessor.
        
        Args:
            chunk_size: Target size for text chunks (default from Config)
            chunk_overlap: Overlap between chunks (default from Config)
        
        Raises:
            ValueError: If chunk_size or chunk_overlap are invalid
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
        
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than "
                f"chunk_size ({chunk_size})"
            )
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.supported_formats = self.SUPPORTED_FORMATS
        
        # Initialize text splitter with intelligent separators
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n\n",  # Multiple line breaks
                "\n\n",    # Paragraph breaks
                "\n",      # Line breaks
                ". ",      # Sentences
                ", ",      # Clauses
                " ",       # Words
                "",        # Characters
            ],
        )
        
        logger.info(
            f"DocumentProcessor initialized with chunk_size={chunk_size}, "
            f"chunk_overlap={chunk_overlap}"
        )
    
    def _detect_framework(self, file_path: Path) -> Optional[str]:
        """
        Detect framework type from file path.
        
        Args:
            file_path: Path to the document
        
        Returns:
            Framework name if detected, None otherwise
        """
        path_str = str(file_path).lower()
        
        for framework, patterns in self.FRAMEWORK_PATTERNS.items():
            if any(pattern in path_str for pattern in patterns):
                return framework
        
        # Check if in policies directory
        if "policies" in path_str:
            return "policy"
        
        return None
    
    def load_pdf(self, file_path: Union[str, Path]) -> List[Dict]:
        """
        Load PDF document.
        
        Args:
            file_path: Path to PDF file
        
        Returns:
            List of document dictionaries with page content and metadata
        
        Raises:
            FileNotFoundError: If file doesn't exist
            Exception: If PDF loading fails
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        logger.info(f"Loading PDF: {file_path}")
        
        try:
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()
            
            logger.info(f"Successfully loaded {len(documents)} pages from {file_path.name}")
            return documents
        
        except Exception as e:
            logger.error(f"Failed to load PDF {file_path}: {e}")
            raise
    
    def load_docx(self, file_path: Union[str, Path]) -> List[Dict]:
        """
        Load DOCX document.
        
        Args:
            file_path: Path to DOCX file
        
        Returns:
            List of document dictionaries with content and metadata
        
        Raises:
            FileNotFoundError: If file doesn't exist
            Exception: If DOCX loading fails
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"DOCX file not found: {file_path}")
        
        logger.info(f"Loading DOCX: {file_path}")
        
        try:
            loader = Docx2txtLoader(str(file_path))
            documents = loader.load()
            
            logger.info(f"Successfully loaded DOCX: {file_path.name}")
            return documents
        
        except Exception as e:
            logger.error(f"Failed to load DOCX {file_path}: {e}")
            raise
    
    def load_txt(self, file_path: Union[str, Path], encoding: str = "utf-8") -> List[Dict]:
        """
        Load TXT document.
        
        Args:
            file_path: Path to TXT file
            encoding: File encoding (default: utf-8)
        
        Returns:
            List of document dictionaries with content and metadata
        
        Raises:
            FileNotFoundError: If file doesn't exist
            Exception: If TXT loading fails
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"TXT file not found: {file_path}")
        
        logger.info(f"Loading TXT: {file_path}")
        
        try:
            loader = TextLoader(str(file_path), encoding=encoding)
            documents = loader.load()
            
            logger.info(f"Successfully loaded TXT: {file_path.name}")
            return documents
        
        except Exception as e:
            logger.error(f"Failed to load TXT {file_path}: {e}")
            raise
    
    def load_document(self, file_path: Union[str, Path]) -> List[Dict]:
        """
        Load document with automatic format detection.
        
        Args:
            file_path: Path to document
        
        Returns:
            List of loaded document chunks
        
        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file doesn't exist
        """
        file_path = Path(file_path)
        file_ext = file_path.suffix.lower()
        
        if file_ext not in self.supported_formats:
            raise ValueError(
                f"Unsupported file format: {file_ext}. "
                f"Supported formats: {self.supported_formats}"
            )
        
        # Route to appropriate loader
        loaders = {
            ".pdf": self.load_pdf,
            ".docx": self.load_docx,
            ".txt": self.load_txt,
        }
        
        return loaders[file_ext](file_path)
    
    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Chunk documents into smaller pieces using RecursiveCharacterTextSplitter.
        
        Args:
            documents: List of document dictionaries from loaders
        
        Returns:
            List of chunked documents with preserved metadata
        """
        logger.info(f"Chunking {len(documents)} documents...")
        
        chunked_docs = self.text_splitter.split_documents(documents)
        
        logger.info(f"Created {len(chunked_docs)} chunks from {len(documents)} documents")
        
        return chunked_docs
    
    def process_document(
        self,
        file_path: Union[str, Path],
        framework: Optional[str] = None,
    ) -> List[ProcessedChunk]:
        """
        Process a single document: load, chunk, and add metadata.
        
        Args:
            file_path: Path to document
            framework: Optional framework override (auto-detected if None)
        
        Returns:
            List of ProcessedChunk objects with content and metadata
        
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported
        """
        file_path = Path(file_path)
        
        logger.info(f"Processing document: {file_path}")
        
        # Load document
        try:
            documents = self.load_document(file_path)
        except Exception as e:
            logger.error(f"Failed to load document {file_path}: {e}")
            raise
        
        # Chunk documents
        chunks = self.chunk_documents(documents)
        
        # Detect framework if not provided
        if framework is None:
            framework = self._detect_framework(file_path)
        
        # Create ProcessedChunk objects with metadata
        processed_chunks = []
        total_chunks = len(chunks)
        
        for idx, chunk in enumerate(chunks):
            # Extract page number if available (PDFs)
            page_number = chunk.metadata.get("page", None)
            
            # Create metadata
            metadata = DocumentMetadata(
                source=str(file_path.absolute()),
                filename=file_path.name,
                file_type=file_path.suffix.lower().lstrip("."),
                framework=framework,
                chunk_id=idx,
                total_chunks=total_chunks,
                page_number=page_number,
                char_count=len(chunk.page_content),
                metadata_extra=chunk.metadata,
            )
            
            # Create processed chunk
            processed_chunk = ProcessedChunk(
                content=chunk.page_content,
                metadata=metadata,
            )
            
            processed_chunks.append(processed_chunk)
        
        logger.info(
            f"Successfully processed {file_path.name}: "
            f"{total_chunks} chunks, framework={framework}"
        )
        
        return processed_chunks
    
    def process_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = True,
        framework: Optional[str] = None,
        show_progress: bool = True,
    ) -> List[ProcessedChunk]:
        """
        Process all supported documents in a directory.
        
        Args:
            directory: Path to directory
            recursive: Process subdirectories (default: True)
            framework: Optional framework override for all files
            show_progress: Show progress bar (default: True)
        
        Returns:
            List of all processed chunks from directory
        
        Raises:
            FileNotFoundError: If directory doesn't exist
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        if not directory.is_dir():
            raise ValueError(f"Path is not a directory: {directory}")
        
        logger.info(f"Processing directory: {directory}")
        
        # Find all supported files
        files = []
        for ext in self.supported_formats:
            if recursive:
                files.extend(directory.rglob(f"*{ext}"))
            else:
                files.extend(directory.glob(f"*{ext}"))
        
        if not files:
            logger.warning(f"No supported files found in {directory}")
            return []
        
        logger.info(f"Found {len(files)} files to process")
        
        # Process files
        all_chunks = []
        
        # Use tqdm for progress tracking
        iterator = tqdm(files, desc="Processing documents") if show_progress else files
        
        for file_path in iterator:
            try:
                chunks = self.process_document(file_path, framework=framework)
                all_chunks.extend(chunks)
                
                if show_progress:
                    iterator.set_postfix({"chunks": len(all_chunks)})
            
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                # Continue processing other files
                continue
        
        logger.info(
            f"Directory processing complete: {len(files)} files, "
            f"{len(all_chunks)} total chunks"
        )
        
        return all_chunks


def test_document_processor():
    """
    Test function for DocumentProcessor.
    
    Tests basic functionality including:
    - Initialization
    - Framework detection
    - Document processing (if test files exist)
    """
    print("=" * 70)
    print("Testing DocumentProcessor")
    print("=" * 70)
    print()
    
    # Test initialization
    print("1. Testing initialization...")
    processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
    print(f"   ✓ Initialized with chunk_size={processor.chunk_size}, "
          f"chunk_overlap={processor.chunk_overlap}")
    print()
    
    # Test framework detection
    print("2. Testing framework detection...")
    test_paths = [
        "data/frameworks/iso27001/test.pdf",
        "data/frameworks/nist_csf/guide.docx",
        "data/policies/security_policy.txt",
    ]
    
    for path in test_paths:
        framework = processor._detect_framework(Path(path))
        print(f"   {path} -> {framework}")
    print()
    
    # Test document processing if files exist
    print("3. Testing document processing...")
    
    # Check for ISO 27001 directory
    iso_dir = Config.FRAMEWORKS_DIR / "iso27001"
    if iso_dir.exists():
        try:
            chunks = processor.process_directory(iso_dir, show_progress=True)
            print(f"   ✓ Processed {len(chunks)} chunks from {iso_dir}")
            
            if chunks:
                print(f"\n   Sample chunk:")
                sample = chunks[0]
                print(f"   - Content length: {len(sample.content)} chars")
                print(f"   - Framework: {sample.metadata.framework}")
                print(f"   - Filename: {sample.metadata.filename}")
                print(f"   - Chunk {sample.metadata.chunk_id + 1}/"
                      f"{sample.metadata.total_chunks}")
        except Exception as e:
            print(f"   ⚠ Could not process directory: {e}")
    else:
        print(f"   ⚠ Test directory not found: {iso_dir}")
        print(f"   Add documents to test processing functionality")
    
    print()
    print("=" * 70)
    print("DocumentProcessor test complete")
    print("=" * 70)


if __name__ == "__main__":
    # Configure logging for standalone testing
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Add parent directory to path for direct execution
    import sys
    from pathlib import Path
    parent_dir = Path(__file__).resolve().parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    
    test_document_processor()

