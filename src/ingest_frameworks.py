"""
Framework Ingestion Script for ISMS RAG Platform
================================================

This script handles the ingestion of security framework documents into the vector store.
It processes documents from ISO 27001, NIST CSF, and CIS Controls frameworks, creating
embeddings and storing them in ChromaDB for retrieval.

Features:
    - Batch processing of multiple frameworks
    - Progress tracking and statistics
    - Interactive user prompts
    - Directory structure creation with READMEs
    - Error handling and recovery
    - Detailed logging

Usage:
    # Ingest all frameworks
    python src/ingest_frameworks.py
    
    # Or use as module
    from src.ingest_frameworks import load_framework, main
    
    # Load specific framework
    stats = load_framework(
        framework_name="iso27001",
        framework_path=Path("data/frameworks/iso27001")
    )

Date: 2026
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

from src.config import Config
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore, RAGRetriever

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Framework information
FRAMEWORKS = {
    "iso27001": {
        "name": "ISO/IEC 27001",
        "description": "Information Security Management System",
        "path": Config.FRAMEWORKS_DIR / "iso27001",
        "priority": 1,
    },
    "nist_csf": {
        "name": "NIST Cybersecurity Framework",
        "description": "Framework for Improving Critical Infrastructure Cybersecurity",
        "path": Config.FRAMEWORKS_DIR / "nist_csf",
        "priority": 2,
    },
    "cis_controls": {
        "name": "CIS Controls",
        "description": "Center for Internet Security Critical Security Controls",
        "path": Config.FRAMEWORKS_DIR / "cis_controls",
        "priority": 3,
    },
}


def create_directory_structure() -> None:
    """
    Create framework directory structure with README files.
    
    Creates:
        - data/frameworks/iso27001/
        - data/frameworks/nist_csf/
        - data/frameworks/cis_controls/
        - data/policies/
        
    Each directory includes a README.md with instructions.
    """
    print("=" * 70)
    print("Creating Framework Directory Structure")
    print("=" * 70)
    print()
    
    # Ensure base directories exist
    Config.ensure_directories()
    
    # Create framework-specific directories with READMEs
    for framework_id, info in FRAMEWORKS.items():
        framework_dir = info["path"]
        framework_dir.mkdir(parents=True, exist_ok=True)
        
        # Create README
        readme_path = framework_dir / "README.md"
        if not readme_path.exists():
            readme_content = f"""# {info['name']}

## Description
{info['description']}

## Instructions
Place your {info['name']} documents in this directory.

Supported formats:
- PDF (.pdf)
- Microsoft Word (.docx)
- Plain text (.txt)

## Example Files
- ISO 27001 Standard.pdf
- Control Objectives.docx
- Implementation Guide.txt

## Processing
Documents in this directory will be:
1. Automatically chunked into smaller segments
2. Converted to embeddings
3. Stored in the vector database
4. Made searchable through the RAG system

Last updated: {datetime.now().strftime("%Y-%m-%d")}
"""
            readme_path.write_text(readme_content, encoding="utf-8")
            print(f"✓ Created {framework_dir / 'README.md'}")
        else:
            print(f"⊙ {framework_dir / 'README.md'} already exists")
    
    # Create policies directory with README
    policies_dir = Config.POLICIES_DIR
    policies_dir.mkdir(parents=True, exist_ok=True)
    
    policies_readme = policies_dir / "README.md"
    if not policies_readme.exists():
        policies_content = """# Organizational Policies

## Description
Store your organization's security and compliance policies here.

## Instructions
Place your policy documents in this directory.

Supported formats:
- PDF (.pdf)
- Microsoft Word (.docx)
- Plain text (.txt)

## Example Files
- Information Security Policy.pdf
- Access Control Policy.docx
- Incident Response Procedure.txt
- Data Classification Policy.pdf

## Processing
Policy documents will be processed and indexed alongside framework documents
for comprehensive compliance queries.

## Categories
Consider organizing policies in subdirectories:
- security/
- privacy/
- compliance/
- operations/
"""
        policies_readme.write_text(policies_content, encoding="utf-8")
        print(f"✓ Created {policies_readme}")
    else:
        print(f"⊙ {policies_readme} already exists")
    
    print()
    print("✓ Directory structure ready")
    print()


def load_framework(
    framework_name: str,
    framework_path: Path,
    vector_store: VectorStore,
    processor: DocumentProcessor,
) -> Dict[str, int]:
    """
    Load a specific framework into the vector store.
    
    Args:
        framework_name: Framework identifier (iso27001, nist_csf, etc.)
        framework_path: Path to framework directory
        vector_store: VectorStore instance
        processor: DocumentProcessor instance
    
    Returns:
        Dictionary with statistics:
            - files_processed: Number of files processed
            - chunks_created: Number of chunks created
            - success: Whether ingestion was successful
    
    Raises:
        FileNotFoundError: If framework directory doesn't exist
    """
    logger.info(f"Loading framework: {framework_name} from {framework_path}")
    
    if not framework_path.exists():
        raise FileNotFoundError(f"Framework directory not found: {framework_path}")
    
    # Get framework info
    framework_info = FRAMEWORKS.get(framework_name, {
        "name": framework_name,
        "description": "Custom framework",
    })
    
    print(f"\n{'=' * 70}")
    print(f"Processing: {framework_info['name']}")
    print(f"{'=' * 70}")
    print(f"Description: {framework_info['description']}")
    print(f"Path: {framework_path}")
    print()
    
    # Check for documents
    supported_extensions = [".pdf", ".docx", ".txt"]
    files = []
    for ext in supported_extensions:
        files.extend(list(framework_path.rglob(f"*{ext}")))
    
    if not files:
        print(f"⚠ No documents found in {framework_path}")
        print(f"  Supported formats: {', '.join(supported_extensions)}")
        return {
            "files_processed": 0,
            "chunks_created": 0,
            "success": False,
        }
    
    print(f"Found {len(files)} document(s)")
    
    # List files
    for i, file in enumerate(files, 1):
        print(f"  {i}. {file.name}")
    print()
    
    # Process documents
    try:
        print("Processing documents...")
        chunks = processor.process_directory(
            framework_path,
            recursive=True,
            framework=framework_name,
            show_progress=True,
        )
        
        print(f"✓ Created {len(chunks)} chunks from {len(files)} file(s)")
        print()
        
        # Add to vector store
        print("Adding to vector store...")
        vector_store.add_documents(chunks, show_progress=True)
        
        print(f"✓ Successfully ingested {framework_info['name']}")
        print()
        
        return {
            "files_processed": len(files),
            "chunks_created": len(chunks),
            "success": True,
        }
    
    except Exception as e:
        logger.error(f"Failed to load framework {framework_name}: {e}")
        print(f"✗ Error: {e}")
        return {
            "files_processed": 0,
            "chunks_created": 0,
            "success": False,
        }


def check_existing_data(vector_store: VectorStore) -> bool:
    """
    Check if vector store already contains data.
    
    Args:
        vector_store: VectorStore instance
    
    Returns:
        True if data exists, False otherwise
    """
    try:
        stats = vector_store.get_collection_stats()
        return stats["total_documents"] > 0
    except Exception:
        return False


def main() -> int:
    """
    Main ingestion workflow.
    
    Workflow:
        1. Create directory structure if needed
        2. Initialize components
        3. Check for existing data
        4. Process each framework
        5. Display final statistics
    
    Returns:
        Exit code (0 = success, 1 = failure)
    """
    print()
    print("=" * 70)
    print("ISMS RAG Platform - Framework Ingestion")
    print("=" * 70)
    print()
    
    # Step 1: Create directory structure
    print("Step 1: Checking directory structure...")
    try:
        create_directory_structure()
    except Exception as e:
        logger.error(f"Failed to create directory structure: {e}")
        print(f"✗ Error: {e}")
        return 1
    
    # Step 2: Initialize components
    print("Step 2: Initializing components...")
    print()
    
    try:
        # Initialize processor
        processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
        )
        print(f"✓ DocumentProcessor initialized")
        print(f"  - Chunk size: {Config.CHUNK_SIZE}")
        print(f"  - Chunk overlap: {Config.CHUNK_OVERLAP}")
        print()
        
        # Initialize vector store
        vector_store = VectorStore(
            collection_name=Config.CHROMA_COLLECTION_NAME,
            persist_directory=Config.CHROMA_PERSIST_DIRECTORY,
        )
        print(f"✓ VectorStore initialized")
        print(f"  - Collection: {Config.CHROMA_COLLECTION_NAME}")
        print(f"  - Embedding model: {Config.EMBEDDING_MODEL}")
        print()
    
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        print(f"✗ Error: {e}")
        return 1
    
    # Step 3: Check for existing data
    print("Step 3: Checking for existing data...")
    
    has_existing_data = check_existing_data(vector_store)
    
    if has_existing_data:
        stats = vector_store.get_collection_stats()
        print(f"⚠ Found existing data in vector store:")
        print(f"  - Total documents: {stats['total_documents']}")
        print(f"  - Frameworks: {stats['frameworks']}")
        print()
        
        # Ask user if they want to continue
        response = input("Continue and add more documents? (y/n): ").strip().lower()
        if response != 'y':
            print("Ingestion cancelled by user")
            return 0
        print()
    else:
        print("✓ No existing data found, starting fresh ingestion")
        print()
    
    # Step 4: Process frameworks
    print("Step 4: Processing frameworks...")
    print()
    
    # Track overall statistics
    overall_stats = {
        "total_files": 0,
        "total_chunks": 0,
        "successful_frameworks": [],
        "failed_frameworks": [],
    }
    
    # Sort frameworks by priority
    sorted_frameworks = sorted(
        FRAMEWORKS.items(),
        key=lambda x: x[1].get("priority", 999)
    )
    
    # Process each framework
    for framework_id, info in sorted_frameworks:
        framework_path = info["path"]
        
        # Check if directory exists and has files
        if not framework_path.exists():
            print(f"⊙ Skipping {info['name']}: directory not found")
            print(f"  Create directory: {framework_path}")
            print()
            continue
        
        # Ask user if they want to process this framework
        print(f"Process {info['name']}?")
        response = input("  (y/n, or 'a' for all remaining): ").strip().lower()
        
        if response == 'n':
            print(f"⊙ Skipped {info['name']}")
            print()
            continue
        
        process_all = (response == 'a')
        
        # Load framework
        try:
            stats = load_framework(
                framework_name=framework_id,
                framework_path=framework_path,
                vector_store=vector_store,
                processor=processor,
            )
            
            if stats["success"]:
                overall_stats["total_files"] += stats["files_processed"]
                overall_stats["total_chunks"] += stats["chunks_created"]
                overall_stats["successful_frameworks"].append(info["name"])
            else:
                overall_stats["failed_frameworks"].append(info["name"])
        
        except Exception as e:
            logger.error(f"Error processing {framework_id}: {e}")
            overall_stats["failed_frameworks"].append(info["name"])
            print(f"✗ Error: {e}")
            print()
        
        # If user selected 'all', don't prompt for remaining frameworks
        if not process_all and framework_id != sorted_frameworks[-1][0]:
            print()
    
    # Step 5: Display final statistics
    print()
    print("=" * 70)
    print("Ingestion Complete")
    print("=" * 70)
    print()
    
    # Get final vector store stats
    final_stats = vector_store.get_collection_stats()
    
    print("Summary:")
    print(f"  Files processed: {overall_stats['total_files']}")
    print(f"  Chunks created: {overall_stats['total_chunks']}")
    print(f"  Total documents in store: {final_stats['total_documents']}")
    print()
    
    if overall_stats["successful_frameworks"]:
        print("✓ Successfully ingested:")
        for fw in overall_stats["successful_frameworks"]:
            print(f"  - {fw}")
        print()
    
    if overall_stats["failed_frameworks"]:
        print("✗ Failed to ingest:")
        for fw in overall_stats["failed_frameworks"]:
            print(f"  - {fw}")
        print()
    
    print("Frameworks in vector store:")
    for fw in final_stats["frameworks"]:
        print(f"  - {fw}")
    print()
    
    # Test search
    print("Testing retrieval...")
    try:
        retriever = RAGRetriever(vector_store)
        test_results = retriever.retrieve("security controls", k=3)
        
        if test_results:
            print(f"✓ Retrieval test successful ({len(test_results)} results)")
            print(f"\nSample result:")
            print(f"  - Framework: {test_results[0].metadata.get('framework')}")
            print(f"  - Source: {test_results[0].metadata.get('filename')}")
            print(f"  - Score: {test_results[0].score:.3f}")
        else:
            print("⚠ No results found (vector store may be empty)")
    except Exception as e:
        print(f"⚠ Retrieval test failed: {e}")
    
    print()
    print("=" * 70)
    print("Ready for queries!")
    print("=" * 70)
    print()
    
    # Return success if at least one framework was processed
    if overall_stats["successful_frameworks"]:
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())

