"""
End-to-End Test for ISMS RAG Platform
Test the complete pipeline from document processing to query answering
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore, RAGRetriever
from src.query_engine import ISMSQueryEngine, ISMSAssistant

def main():
    print("=" * 80)
    print("ISMS RAG Platform - End-to-End Test")
    print("=" * 80)
    print()
    
    # Step 1: Process documents
    print("Step 1: Processing documents...")
    processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
    test_file = "data/frameworks/tests/test_iso27001.txt"
    
    chunks = processor.process_document(test_file)
    print(f"   -> Processed {len(chunks)} chunks from {test_file}")
    print()
    
    # Step 2: Initialize vector store
    print("Step 2: Creating vector store...")
    vector_store = VectorStore(
        collection_name="isms_demo",
        persist_directory="./demo_chroma_db"
    )
    print(f"   -> Vector store initialized")
    print()
    
    # Step 3: Add documents
    print("Step 3: Adding documents to vector store...")
    vector_store.add_documents(chunks, show_progress=False)
    
    stats = vector_store.get_collection_stats()
    print(f"   -> Added {stats['total_documents']} documents")
    print(f"   - Frameworks: {stats['frameworks']}")
    print(f"   - File types: {stats['file_types']}")
    print()
    
    # Step 4: Initialize RAG engine
    print("Step 4: Initializing RAG query engine...")
    retriever = RAGRetriever(vector_store)
    engine = ISMSQueryEngine(retriever=retriever, language="fr")
    assistant = ISMSAssistant(engine)
    print(f"   -> Query engine ready (model: {engine.model})")
    print()
    
    # Step 5: Test queries
    print("=" * 80)
    print("Testing Queries")
    print("=" * 80)
    print()
    
    # French query
    print("Query 1 (French):")
    print("-" * 80)
    question_fr = "Quels sont les contrôles de sécurité physique dans ISO 27001 ?"
    print(f"Question: {question_fr}")
    print()
    
    result = assistant.ask_with_sources(question_fr, framework="iso27001")
    print(f"Answer:\n{result['answer']}")
    print(f"\nSources used: {result['context_used']}")
    for i, source in enumerate(result['sources'], 1):
        print(f"  [{i}] {source['filename']} (score: {source['score']:.3f})")
    print()
    
    # English query
    print("Query 2 (English):")
    print("-" * 80)
    question_en = "What are the access control requirements?"
    print(f"Question: {question_en}")
    print()
    
    answer_en = assistant.ask(question_en, framework="iso27001", language="en")
    print(f"Answer:\n{answer_en}")
    print()
    
    # Control explanation
    print("Query 3 (Control Explanation):")
    print("-" * 80)
    print("Question: Explain control A.7.1")
    print()
    
    explanation = assistant.explain_control("A.7.1", framework="iso27001", language="en")
    print(f"Answer:\n{explanation}")
    print()
    
    # Cleanup
    print("=" * 80)
    print("Cleanup...")
    vector_store.delete_collection()
    
    import shutil
    demo_dir = Path("./demo_chroma_db")
    if demo_dir.exists():
        shutil.rmtree(demo_dir, ignore_errors=True)
    print("   -> Test database removed")
    print()
    
    print("=" * 80)
    print("End-to-End Test Complete!")
    print("=" * 80)

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise
    
    main()
