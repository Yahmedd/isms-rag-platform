#!/usr/bin/env python3
"""
Setup Verification Script for ISMS RAG Platform
===============================================

This script verifies that the ISMS RAG platform is correctly configured
and ready to use. It checks Python version, dependencies, environment
configuration, directories, and database setup.

Features:
    - Python version verification
    - Environment file validation
    - Dependency checking
    - Directory structure verification
    - Framework document detection
    - Vector database testing
    - Basic query testing
    - Actionable next steps

Usage:
    python verify_setup.py

Date: 2026
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict


# Status symbols
SYMBOLS = {
    "success": "✓",
    "error": "✗",
    "warning": "⚠",
    "info": "ℹ",
}


def print_status(message: str, status: str = "info", indent: int = 0) -> None:
    """
    Print a status message with symbol.
    
    Args:
        message: Message to display
        status: Status type (success, error, warning, info)
        indent: Number of spaces to indent
    """
    symbol = SYMBOLS.get(status, "•")
    prefix = " " * indent
    print(f"{prefix}{symbol} {message}")


def print_header(title: str, width: int = 70) -> None:
    """Print a formatted header."""
    print()
    print("=" * width)
    print(title.center(width))
    print("=" * width)
    print()


def check_python_version() -> Tuple[bool, str]:
    """
    Check if Python version meets requirements.
    
    Returns:
        Tuple of (success, message)
    """
    required_version = (3, 9)
    current_version = sys.version_info[:2]
    
    if current_version >= required_version:
        return True, f"Python {current_version[0]}.{current_version[1]}"
    else:
        return False, f"Python {current_version[0]}.{current_version[1]} (requires {required_version[0]}.{required_version[1]}+)"


def check_env_file() -> Tuple[bool, str, List[str]]:
    """
    Check if .env file exists and has required variables.
    
    Returns:
        Tuple of (success, message, missing_vars)
    """
    env_file = Path(".env")
    template_file = Path(".env.template")
    
    if not env_file.exists():
        if template_file.exists():
            return False, ".env file missing (template exists)", []
        else:
            return False, ".env and .env.template both missing", []
    
    # Check for required variables
    required_vars = [
        "LLM_MODEL",
        "EMBEDDING_MODEL",
        "CHROMA_PERSIST_DIRECTORY",
        "CHUNK_SIZE",
    ]
    
    missing_vars = []
    try:
        with open(env_file, 'r', encoding='utf-8') as f:
            content = f.read()
            for var in required_vars:
                if f"{var}=" not in content:
                    missing_vars.append(var)
        
        if missing_vars:
            return False, f".env exists but missing {len(missing_vars)} variable(s)", missing_vars
        else:
            return True, ".env file configured", []
    
    except Exception as e:
        return False, f"Error reading .env: {e}", []


def check_dependencies() -> Tuple[bool, str, List[str]]:
    """
    Check if required Python packages are installed.
    
    Returns:
        Tuple of (success, message, missing_packages)
    """
    required_packages = [
        ("langchain", "langchain"),
        ("langchain_community", "langchain-community"),
        ("langchain_text_splitters", "langchain-text-splitters"),
        ("chromadb", "chromadb"),
        ("sentence_transformers", "sentence-transformers"),
        ("openai", "openai"),
        ("pypdf", "pypdf"),
        ("docx", "python-docx"),
        ("tqdm", "tqdm"),
        ("dotenv", "python-dotenv"),
    ]
    
    missing = []
    for module_name, package_name in required_packages:
        try:
            __import__(module_name)
        except ImportError:
            missing.append(package_name)
    
    if missing:
        return False, f"{len(missing)} package(s) missing", missing
    else:
        return True, f"All {len(required_packages)} packages installed", []


def check_directories() -> Tuple[bool, str, List[str]]:
    """
    Check if required directories exist.
    
    Returns:
        Tuple of (success, message, missing_dirs)
    """
    required_dirs = [
        "src",
        "data",
        "data/frameworks",
        "data/frameworks/iso27001",
        "data/frameworks/nist_csf",
        "data/frameworks/cis_controls",
        "data/policies",
        "tests",
    ]
    
    missing = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing.append(dir_path)
    
    if missing:
        return False, f"{len(missing)} director(ies) missing", missing
    else:
        return True, f"All {len(required_dirs)} directories exist", []


def check_framework_documents() -> Tuple[bool, str, Dict[str, int]]:
    """
    Check for framework documents.
    
    Returns:
        Tuple of (success, message, document_counts)
    """
    frameworks = {
        "ISO 27001": Path("data/frameworks/iso27001"),
        "NIST CSF": Path("data/frameworks/nist_csf"),
        "CIS Controls": Path("data/frameworks/cis_controls"),
    }
    
    counts = {}
    total = 0
    
    for name, path in frameworks.items():
        if path.exists():
            doc_files = list(path.rglob("*.pdf")) + list(path.rglob("*.docx")) + list(path.rglob("*.txt"))
            # Exclude README files
            doc_files = [f for f in doc_files if f.name.lower() != "readme.md"]
            count = len(doc_files)
            counts[name] = count
            total += count
        else:
            counts[name] = 0
    
    if total == 0:
        return False, "No framework documents found", counts
    else:
        return True, f"{total} document(s) found", counts


def check_vector_database() -> Tuple[bool, str]:
    """
    Check if vector database is initialized.
    
    Returns:
        Tuple of (success, message)
    """
    try:
        # Try to import and check
        sys.path.insert(0, str(Path(__file__).parent))
        from src.config import Config
        from src.vector_store import VectorStore
        
        chroma_dir = Path(Config.CHROMA_PERSIST_DIRECTORY)
        
        if not chroma_dir.exists():
            return False, "Vector database not initialized"
        
        # Try to access the vector store
        try:
            vector_store = VectorStore(
                collection_name=Config.CHROMA_COLLECTION_NAME,
                persist_directory=Config.CHROMA_PERSIST_DIRECTORY,
            )
            stats = vector_store.get_collection_stats()
            doc_count = stats.get("total_documents", 0)
            
            if doc_count == 0:
                return False, "Vector database empty (0 documents)"
            else:
                return True, f"Vector database initialized ({doc_count} documents)"
        
        except Exception as e:
            return False, f"Error accessing database: {str(e)[:50]}"
    
    except ImportError as e:
        return False, f"Cannot import modules: {str(e)[:50]}"
    except Exception as e:
        return False, f"Error: {str(e)[:50]}"


def test_basic_query() -> Tuple[bool, str]:
    """
    Test if basic query works.
    
    Returns:
        Tuple of (success, message)
    """
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from src.config import Config
        from src.vector_store import VectorStore, RAGRetriever
        from src.query_engine import ISMSQueryEngine
        
        # Initialize components
        vector_store = VectorStore(
            collection_name=Config.CHROMA_COLLECTION_NAME,
            persist_directory=Config.CHROMA_PERSIST_DIRECTORY,
        )
        
        stats = vector_store.get_collection_stats()
        if stats.get("total_documents", 0) == 0:
            return False, "Cannot test - no documents in database"
        
        # Try a simple retrieval
        retriever = RAGRetriever(vector_store)
        results = retriever.retrieve("security", k=1)
        
        if results:
            return True, "Query test successful"
        else:
            return False, "Query returned no results"
    
    except ImportError as e:
        return False, f"Import error: {str(e)[:30]}"
    except Exception as e:
        return False, f"Error: {str(e)[:30]}"


def main() -> int:
    """
    Run all verification checks.
    
    Returns:
        Exit code (0 = all checks passed, 1 = some checks failed)
    """
    print_header("ISMS RAG Platform - Setup Verification")
    
    results = {}
    
    # Check 1: Python version
    print("1. Checking Python version...")
    success, message = check_python_version()
    results["Python Version"] = success
    print_status(message, "success" if success else "error", indent=2)
    print()
    
    # Check 2: Environment file
    print("2. Checking environment configuration...")
    success, message, missing = check_env_file()
    results["Environment File"] = success
    print_status(message, "success" if success else "error", indent=2)
    if missing:
        for var in missing:
            print_status(f"Missing: {var}", "warning", indent=4)
    print()
    
    # Check 3: Dependencies
    print("3. Checking Python dependencies...")
    success, message, missing = check_dependencies()
    results["Dependencies"] = success
    print_status(message, "success" if success else "error", indent=2)
    if missing:
        print_status("Missing packages:", "warning", indent=4)
        for pkg in missing[:5]:  # Show first 5
            print_status(pkg, "info", indent=6)
        if len(missing) > 5:
            print_status(f"... and {len(missing) - 5} more", "info", indent=6)
    print()
    
    # Check 4: Directories
    print("4. Checking directory structure...")
    success, message, missing = check_directories()
    results["Directories"] = success
    print_status(message, "success" if success else "error", indent=2)
    if missing:
        for dir_path in missing:
            print_status(f"Missing: {dir_path}", "warning", indent=4)
    print()
    
    # Check 5: Framework documents
    print("5. Checking framework documents...")
    success, message, counts = check_framework_documents()
    results["Framework Documents"] = success
    print_status(message, "success" if success else "warning", indent=2)
    for framework, count in counts.items():
        status = "success" if count > 0 else "warning"
        print_status(f"{framework}: {count} document(s)", status, indent=4)
    print()
    
    # Check 6: Vector database
    print("6. Checking vector database...")
    success, message = check_vector_database()
    results["Vector Database"] = success
    print_status(message, "success" if success else "warning", indent=2)
    print()
    
    # Check 7: Basic query test
    print("7. Testing basic query...")
    success, message = test_basic_query()
    results["Query Test"] = success
    print_status(message, "success" if success else "warning", indent=2)
    print()
    
    # Summary
    print_header("Verification Summary")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"Checks Passed: {passed}/{total}")
    print()
    
    if passed == total:
        print_status("All checks passed! Platform is ready to use.", "success")
        print()
        print("Next steps:")
        print("  1. Run demo: python src/demo.py")
        print("  2. Add more documents: python src/ingest_frameworks.py")
        print()
        return 0
    else:
        print_status(f"{total - passed} check(s) failed", "error")
        print()
        print("Recommended actions:")
        print()
        
        if not results.get("Python Version"):
            print("  • Upgrade Python to version 3.9 or higher")
        
        if not results.get("Environment File"):
            print("  • Copy .env.template to .env and configure settings")
            print("    Command: cp .env.template .env")
        
        if not results.get("Dependencies"):
            print("  • Install missing packages:")
            print("    Command: pip install -r requirements.txt")
        
        if not results.get("Directories"):
            print("  • Create missing directories:")
            print("    Command: python setup_project.sh")
        
        if not results.get("Framework Documents"):
            print("  • Add framework documents to data/frameworks/")
            print("    - Place PDF, DOCX, or TXT files in respective framework folders")
        
        if not results.get("Vector Database"):
            print("  • Initialize vector database:")
            print("    Command: python src/ingest_frameworks.py")
        
        if not results.get("Query Test"):
            print("  • Ensure documents are ingested and database is populated")
        
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
