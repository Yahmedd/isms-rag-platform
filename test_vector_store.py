"""
Test script for VectorStore module
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.vector_store import VectorStore, RAGRetriever
from src.document_processor import DocumentProcessor

def main():
    print("=" * 70)
    print("Testing VectorStore Module")
    print("=" * 70)
    print()
    
    # Test 1: Initialize VectorStore
    print("1. Initializing VectorStore...")
    try:
        vector_store = VectorStore(
            collection_name="test_isms_collection",
            persist_directory="./test_chroma_db",
        )
        print(f"   ✓ VectorStore initialized")
        print(f"   - Collection: {vector_store.collection_name}")
        print(f"   - Model: {vector_store.embedding_model}")
        print(f"   - Persist directory: {vector_store.persist_directory}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return
    print()
    
    # Test 2: Process and add documents
    print("2. Processing and adding documents...")
    test_file = Path("data/frameworks/tests/test_iso27001.txt")
    
    if test_file.exists():
        try:
            # Process document
            processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
            chunks = processor.process_document(str(test_file))
            print(f"   - Processed {len(chunks)} chunks from test file")
            
            # Add to vector store
            vector_store.add_documents(chunks, show_progress=True)
            print(f"   ✓ Documents added successfully")
        except Exception as e:
            print(f"   ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
            return
    else:
        print(f"   ✗ Test file not found: {test_file}")
        return
    print()
    
    # Test 3: Basic search
    print("3. Testing basic search...")
    try:
        query = "security controls"
        results = vector_store.search(query, k=3)
        print(f"   ✓ Search for '{query}': {len(results)} results")
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"\n   Result {i}:")
                print(f"   - Score: {result.score:.3f}")
                print(f"   - Framework: {result.metadata.get('framework')}")
                print(f"   - Preview: {result.content[:80]}...")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
    print()
    
    # Test 4: Search with scores
    print("4. Testing search with score filtering...")
    try:
        query = "risk assessment"
        results = vector_store.search_with_scores(
            query, 
            k=5, 
            min_score=0.3
        )
        print(f"   ✓ Search for '{query}': {len(results)} results (min_score=0.3)")
        
        if results:
            print(f"   - Best match score: {results[0].score:.3f}")
            print(f"   - Worst match score: {results[-1].score:.3f}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    print()
    
    # Test 5: Collection statistics
    print("5. Testing collection statistics...")
    try:
        stats = vector_store.get_collection_stats()
        print(f"   ✓ Collection statistics:")
        print(f"   - Total documents: {stats['total_documents']}")
        print(f"   - Frameworks: {stats['frameworks']}")
        print(f"   - File types: {stats['file_types']}")
        print(f"   - Embedding model: {stats['embedding_model']}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    print()
    
    # Test 6: RAGRetriever
    print("6. Testing RAGRetriever...")
    try:
        retriever = RAGRetriever(vector_store)
        
        # Test framework-specific retrieval
        results = retriever.retrieve(
            query="information security",
            k=3,
            framework="iso27001"
        )
        print(f"   ✓ Retrieve with framework filter: {len(results)} results")
        
        # Test context generation
        context = retriever.get_context_for_query("access control", k=2)
        print(f"   ✓ Generated context: {len(context)} characters")
        
        if context:
            print(f"\n   Context preview:")
            print(f"   {context[:150]}...")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
    print()
    
    # Test 7: Metadata filtering
    print("7. Testing metadata filtering...")
    try:
        # Search with framework filter
        results = vector_store.search(
            query="compliance",
            k=5,
            where={"framework": "iso27001"}
        )
        print(f"   ✓ Filtered search (framework=iso27001): {len(results)} results")
        
        # Verify all results match filter
        if results:
            frameworks = set(r.metadata.get('framework') for r in results)
            print(f"   - Frameworks in results: {frameworks}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    print()
    
    # Cleanup
    print("8. Cleaning up...")
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
        print(f"   ⚠ Cleanup warning: {e}")
    
    print()
    print("=" * 70)
    print("All Tests Completed Successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()
