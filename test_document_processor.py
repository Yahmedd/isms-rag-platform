"""
Test script for DocumentProcessor module
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.document_processor import DocumentProcessor, DocumentMetadata, ProcessedChunk
from src.config import Config

def main():
    print("=" * 70)
    print("Testing DocumentProcessor")
    print("=" * 70)
    print()
    
    # Initialize processor
    print("1. Initializing DocumentProcessor...")
    processor = DocumentProcessor(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    )
    print(f"   ✓ Chunk size: {processor.chunk_size}")
    print(f"   ✓ Chunk overlap: {processor.chunk_overlap}")
    print(f"   ✓ Supported formats: {processor.SUPPORTED_FORMATS}")
    print()
    
    # Test with the test file
    test_file = "data/frameworks/tests/test_iso27001.txt"
    print(f"2. Testing single document processing...")
    print(f"   File: {test_file}")
    
    if Path(test_file).exists():
        try:
            chunks = processor.process_document(test_file)
            print(f"   ✓ Successfully processed: {len(chunks)} chunks")
            print()
            
            # Display detailed information about chunks
            print("3. Chunk Analysis:")
            for i, chunk in enumerate(chunks, 1):
                print(f"\n   Chunk {i}/{len(chunks)}:")
                print(f"   - Content length: {chunk.metadata.char_count} characters")
                print(f"   - Framework: {chunk.metadata.framework}")
                print(f"   - Source: {chunk.metadata.source}")
                print(f"   - File type: {chunk.metadata.file_type}")
                print(f"   - Content preview: {chunk.content[:100]}...")
            print()
            
        except Exception as e:
            print(f"   ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"   ✗ Test file not found: {test_file}")
        print(f"   Create a test file first")
    
    # Test directory processing
    print("\n4. Testing directory processing...")
    test_dir = Path("data/frameworks/tests")
    
    if test_dir.exists():
        try:
            all_chunks = processor.process_directory(test_dir, show_progress=True)
            print(f"   ✓ Processed {len(all_chunks)} total chunks from directory")
            
            # Group by framework
            frameworks = {}
            for chunk in all_chunks:
                fw = chunk.metadata.framework or "unknown"
                frameworks[fw] = frameworks.get(fw, 0) + 1
            
            print(f"\n   Chunks by framework:")
            for fw, count in frameworks.items():
                print(f"   - {fw}: {count} chunks")
            
        except Exception as e:
            print(f"   ✗ Error: {e}")
    else:
        print(f"   ⚠ Directory not found: {test_dir}")
    
    print()
    print("=" * 70)
    print("Test Complete")
    print("=" * 70)

if __name__ == "__main__":
    main()
