"""
Vector Store Management for ISMS RAG Platform
=============================================

This module provides vector database operations for the ISMS RAG platform using
ChromaDB for persistent storage and HuggingFace embeddings for multilingual support.

The module includes:
- VectorStore: Low-level ChromaDB operations
- RAGRetriever: High-level retrieval with advanced filtering
- Batch processing for efficient ingestion
- Metadata-based filtering and search

Features:
    - ChromaDB integration with persistent storage
    - HuggingFace multilingual embeddings
    - Framework and language filtering
    - Similarity search with scoring
    - Collection management and statistics
    - Batch processing for large datasets

Usage:
    from src.config import Config
    from src.vector_store import VectorStore, RAGRetriever
    from src.document_processor import DocumentProcessor
    
    # Initialize vector store
    vector_store = VectorStore(
        collection_name=Config.CHROMA_COLLECTION_NAME,
        persist_directory=Config.CHROMA_PERSIST_DIRECTORY
    )
    
    # Process and add documents
    processor = DocumentProcessor()
    chunks = processor.process_document("data/frameworks/iso27001/doc.pdf")
    vector_store.add_documents(chunks)
    
    # Search
    results = vector_store.search("access control requirements", k=5)
    
    # Advanced retrieval
    retriever = RAGRetriever(vector_store)
    results = retriever.retrieve(
        query="ISO 27001 requirements",
        framework="iso27001",
        k=5
    )

Date: 2026
"""

import logging
import uuid
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.config import Config

# Configure logger
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """
    Container for search results.
    
    Attributes:
        content: Text content of the chunk
        metadata: Associated metadata
        score: Similarity score (0.0-1.0)
        rank: Result rank (0 = best match)
    """
    content: str
    metadata: Dict[str, Any]
    score: float
    rank: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "content": self.content,
            "metadata": self.metadata,
            "score": self.score,
            "rank": self.rank,
        }
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        framework = self.metadata.get("framework", "unknown")
        source = self.metadata.get("filename", "unknown")
        return (
            f"SearchResult(rank={self.rank}, score={self.score:.3f}, "
            f"framework={framework}, source={source})"
        )


class VectorStore:
    """
    ChromaDB vector store wrapper for document embeddings.
    
    This class provides low-level operations for storing and retrieving
    document embeddings using ChromaDB with HuggingFace embeddings.
    
    Attributes:
        collection_name: Name of the ChromaDB collection
        persist_directory: Directory for ChromaDB persistence
        embedding_model: Name of the HuggingFace embedding model
        client: ChromaDB client instance
        collection: ChromaDB collection instance
    """
    
    def __init__(
        self,
        collection_name: str = Config.CHROMA_COLLECTION_NAME,
        persist_directory: str = Config.CHROMA_PERSIST_DIRECTORY,
        embedding_model: str = Config.EMBEDDING_MODEL,
    ):
        """
        Initialize VectorStore.
        
        Args:
            collection_name: Name for the ChromaDB collection
            persist_directory: Directory for persistent storage
            embedding_model: HuggingFace model for embeddings
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        
        # Create persist directory if it doesn't exist
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client with persistent storage
        logger.info(f"Initializing ChromaDB client at {persist_directory}")
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
            )
        )
        
        # Initialize SentenceTransformer model for embeddings
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model_instance = SentenceTransformer(embedding_model)
        
        # Create custom embedding function
        class CustomEmbeddingFunction:
            def __init__(self, model, model_name):
                self.model = model
                self.model_name = model_name
            
            def __call__(self, input: List[str]) -> List[List[float]]:
                embeddings = self.model.encode(input, convert_to_numpy=True)
                return embeddings.tolist()
            
            def embed_query(self, input: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
                """Embed query string(s)."""
                # Input can be a string or list of strings
                if isinstance(input, str):
                    embedding = self.model.encode([input], convert_to_numpy=True)[0]
                    return embedding.tolist()
                else:
                    embeddings = self.model.encode(input, convert_to_numpy=True)
                    return embeddings.tolist()
            
            def name(self) -> str:
                """Return the name of the embedding function."""
                return self.model_name
        
        self.embedding_function = CustomEmbeddingFunction(self.embedding_model_instance, embedding_model)
        
        # Get or create collection with cosine similarity
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
            )
            logger.info(f"Loaded existing collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={
                    "description": "ISMS framework documents",
                    "hnsw:space": "cosine",  # Use cosine similarity
                },
            )
            logger.info(f"Created new collection: {collection_name}")
    
    def add_documents(
        self,
        documents: List[Any],
        batch_size: int = 100,
        show_progress: bool = True,
    ) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of ProcessedChunk objects from DocumentProcessor
            batch_size: Number of documents to process in each batch
            show_progress: Show progress bar
        
        Raises:
            ValueError: If documents list is empty
            Exception: If adding documents fails
        """
        if not documents:
            raise ValueError("Documents list cannot be empty")
        
        logger.info(f"Adding {len(documents)} documents to vector store")
        
        # Process in batches
        num_batches = (len(documents) + batch_size - 1) // batch_size
        
        iterator = range(num_batches)
        if show_progress:
            iterator = tqdm(iterator, desc="Adding documents", unit="batch")
        
        total_added = 0
        
        for batch_idx in iterator:
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(documents))
            batch = documents[start_idx:end_idx]
            
            # Prepare batch data
            ids = []
            contents = []
            metadatas = []
            
            for doc in batch:
                # Generate unique ID
                doc_id = str(uuid.uuid4())
                ids.append(doc_id)
                
                # Extract content
                contents.append(doc.content)
                
                # Prepare metadata (ChromaDB requires strings for all values)
                metadata = {
                    "source": doc.metadata.source,
                    "filename": doc.metadata.filename,
                    "file_type": doc.metadata.file_type,
                    "framework": doc.metadata.framework or "unknown",
                    "chunk_id": str(doc.metadata.chunk_id),
                    "total_chunks": str(doc.metadata.total_chunks),
                    "char_count": str(doc.metadata.char_count),
                }
                
                # Add page number if available
                if doc.metadata.page_number is not None:
                    metadata["page_number"] = str(doc.metadata.page_number)
                
                metadatas.append(metadata)
            
            # Add batch to collection
            try:
                self.collection.add(
                    ids=ids,
                    documents=contents,
                    metadatas=metadatas,
                )
                total_added += len(batch)
                
                if show_progress:
                    iterator.set_postfix({"total": total_added})
            
            except Exception as e:
                logger.error(f"Failed to add batch {batch_idx}: {e}")
                raise
        
        logger.info(f"Successfully added {total_added} documents to {self.collection_name}")
    
    def search(
        self,
        query: str,
        k: int = Config.TOP_K_RESULTS,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            where: Metadata filter (e.g., {"framework": "iso27001"})
        
        Returns:
            List of SearchResult objects
        
        Raises:
            ValueError: If k <= 0
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        
        logger.info(f"Searching for: '{query}' (k={k}, filter={where})")
        
        # Query collection
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=where,
        )
        
        # Parse results
        search_results = []
        
        if results["documents"] and results["documents"][0]:
            for rank, (content, metadata, distance) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )):
                # Convert distance to similarity score
                # ChromaDB with cosine returns distance = 1 - cosine_similarity
                # So similarity = 1 - distance
                score = max(0.0, min(1.0, 1.0 - distance))
                
                search_result = SearchResult(
                    content=content,
                    metadata=metadata,
                    score=score,
                    rank=rank,
                )
                
                search_results.append(search_result)
        
        logger.info(f"Found {len(search_results)} results")
        
        return search_results
    
    def search_with_scores(
        self,
        query: str,
        k: int = Config.TOP_K_RESULTS,
        min_score: float = Config.SIMILARITY_THRESHOLD,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Search with score filtering.
        
        Args:
            query: Search query
            k: Maximum number of results
            min_score: Minimum similarity score (0.0-1.0)
            where: Metadata filter
        
        Returns:
            List of SearchResult objects with score >= min_score
        """
        results = self.search(query, k=k, where=where)
        
        # Filter by minimum score
        filtered_results = [r for r in results if r.score >= min_score]
        
        logger.info(
            f"Filtered {len(results)} results to {len(filtered_results)} "
            f"with min_score={min_score}"
        )
        
        return filtered_results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        count = self.collection.count()
        
        # Get sample to analyze metadata
        sample = self.collection.peek(limit=min(count, 100))
        
        # Analyze frameworks
        frameworks = set()
        file_types = set()
        
        if sample["metadatas"]:
            for metadata in sample["metadatas"]:
                frameworks.add(metadata.get("framework", "unknown"))
                file_types.add(metadata.get("file_type", "unknown"))
        
        stats = {
            "collection_name": self.collection_name,
            "total_documents": count,
            "frameworks": sorted(list(frameworks)),
            "file_types": sorted(list(file_types)),
            "embedding_model": self.embedding_model,
            "persist_directory": self.persist_directory,
        }
        
        return stats
    
    def delete_collection(self) -> None:
        """
        Delete the entire collection.
        
        Warning: This operation cannot be undone!
        """
        logger.warning(f"Deleting collection: {self.collection_name}")
        
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Collection {self.collection_name} deleted successfully")
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise
    
    def clear_collection(self) -> None:
        """
        Clear all documents from the collection without deleting it.
        
        This recreates the collection, effectively removing all documents.
        """
        logger.warning(f"Clearing collection: {self.collection_name}")
        
        # Delete and recreate collection
        self.delete_collection()
        
        self.collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"description": "ISMS framework documents"},
        )
        
        logger.info(f"Collection {self.collection_name} cleared successfully")


class RAGRetriever:
    """
    High-level retrieval interface with advanced filtering.
    
    This class provides convenient methods for retrieving documents with
    framework-specific filtering, language support, and result ranking.
    
    Attributes:
        vector_store: VectorStore instance
    """
    
    def __init__(self, vector_store: VectorStore):
        """
        Initialize RAGRetriever.
        
        Args:
            vector_store: VectorStore instance
        """
        self.vector_store = vector_store
        logger.info("RAGRetriever initialized")
    
    def retrieve(
        self,
        query: str,
        k: int = Config.TOP_K_RESULTS,
        framework: Optional[str] = None,
        min_score: float = Config.SIMILARITY_THRESHOLD,
    ) -> List[SearchResult]:
        """
        Retrieve relevant documents with optional framework filtering.
        
        Args:
            query: Search query
            k: Number of results to return
            framework: Optional framework filter (iso27001, nist_csf, cis_controls)
            min_score: Minimum similarity score
        
        Returns:
            List of SearchResult objects
        """
        # Build metadata filter
        where = None
        if framework:
            where = {"framework": framework}
        
        # Search with filtering
        results = self.vector_store.search_with_scores(
            query=query,
            k=k,
            min_score=min_score,
            where=where,
        )
        
        logger.info(
            f"Retrieved {len(results)} documents "
            f"(framework={framework}, min_score={min_score})"
        )
        
        return results
    
    def retrieve_by_framework(
        self,
        query: str,
        framework: str,
        k: int = Config.TOP_K_RESULTS,
    ) -> List[SearchResult]:
        """
        Retrieve documents from a specific framework.
        
        Args:
            query: Search query
            framework: Framework name (iso27001, nist_csf, cis_controls, policy)
            k: Number of results
        
        Returns:
            List of SearchResult objects from the specified framework
        """
        return self.retrieve(query=query, k=k, framework=framework)
    
    def retrieve_multi_framework(
        self,
        query: str,
        frameworks: List[str],
        k_per_framework: int = 3,
    ) -> Dict[str, List[SearchResult]]:
        """
        Retrieve documents from multiple frameworks.
        
        Args:
            query: Search query
            frameworks: List of framework names
            k_per_framework: Number of results per framework
        
        Returns:
            Dictionary mapping framework names to search results
        """
        results = {}
        
        for framework in frameworks:
            framework_results = self.retrieve_by_framework(
                query=query,
                framework=framework,
                k=k_per_framework,
            )
            results[framework] = framework_results
        
        return results
    
    def get_context_for_query(
        self,
        query: str,
        k: int = 5,
        framework: Optional[str] = None,
    ) -> str:
        """
        Get concatenated context from top results for RAG.
        
        Args:
            query: Search query
            k: Number of results to include
            framework: Optional framework filter
        
        Returns:
            Concatenated text from top results with source citations
        """
        results = self.retrieve(query=query, k=k, framework=framework)
        
        if not results:
            return ""
        
        # Build context with citations
        context_parts = []
        
        for result in results:
            source = result.metadata.get("filename", "unknown")
            framework_name = result.metadata.get("framework", "unknown")
            
            citation = f"[Source: {source} ({framework_name})]"
            context_parts.append(f"{citation}\n{result.content}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        return context


def test_vector_store():
    """
    Test function for VectorStore and RAGRetriever.
    
    Tests:
    - VectorStore initialization
    - Document addition
    - Search functionality
    - Collection statistics
    - RAGRetriever operations
    """
    print("=" * 70)
    print("Testing VectorStore")
    print("=" * 70)
    print()
    
    # Test initialization
    print("1. Testing VectorStore initialization...")
    try:
        vector_store = VectorStore(
            collection_name="test_collection",
            persist_directory="./test_chroma_db",
        )
        print(f"   ✓ VectorStore initialized")
        print(f"   - Collection: {vector_store.collection_name}")
        print(f"   - Model: {vector_store.embedding_model}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return
    print()
    
    # Test with processed documents if available
    print("2. Testing document addition...")
    test_file = Path("data/frameworks/tests/test_iso27001.txt")
    
    if test_file.exists():
        try:
            from src.document_processor import DocumentProcessor
            
            processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
            chunks = processor.process_document(test_file)
            
            print(f"   - Processed {len(chunks)} chunks")
            
            # Add to vector store
            vector_store.add_documents(chunks, show_progress=True)
            print(f"   ✓ Documents added successfully")
        except Exception as e:
            print(f"   ✗ Error: {e}")
    else:
        print(f"   ⚠ Test file not found: {test_file}")
    print()
    
    # Test search
    print("3. Testing search...")
    try:
        results = vector_store.search("security controls", k=3)
        print(f"   ✓ Found {len(results)} results")
        
        if results:
            print(f"\n   Top result:")
            print(f"   - Score: {results[0].score:.3f}")
            print(f"   - Framework: {results[0].metadata.get('framework')}")
            print(f"   - Content: {results[0].content[:100]}...")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    print()
    
    # Test collection stats
    print("4. Testing collection statistics...")
    try:
        stats = vector_store.get_collection_stats()
        print(f"   ✓ Collection statistics:")
        print(f"   - Total documents: {stats['total_documents']}")
        print(f"   - Frameworks: {stats['frameworks']}")
        print(f"   - File types: {stats['file_types']}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    print()
    
    # Test RAGRetriever
    print("5. Testing RAGRetriever...")
    try:
        retriever = RAGRetriever(vector_store)
        
        results = retriever.retrieve("risk assessment", k=3, framework="iso27001")
        print(f"   ✓ Retrieved {len(results)} results with framework filter")
        
        context = retriever.get_context_for_query("access control", k=2)
        print(f"   ✓ Generated context: {len(context)} characters")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    print()
    
    # Cleanup
    print("6. Cleaning up test collection...")
    try:
        vector_store.delete_collection()
        print(f"   ✓ Test collection deleted")
        
        # Remove test directory
        import shutil
        test_dir = Path("./test_chroma_db")
        if test_dir.exists():
            shutil.rmtree(test_dir)
            print(f"   ✓ Test directory removed")
    except Exception as e:
        print(f"   ⚠ Cleanup error: {e}")
    
    print()
    print("=" * 70)
    print("VectorStore test complete")
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
    
    test_vector_store()

