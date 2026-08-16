"""
Interactive Demo for ISMS RAG Platform
======================================

This script provides an interactive demonstration of the ISMS RAG platform,
allowing users to query the security framework knowledge base in both French
and English.

Features:
    - Interactive question mode
    - Demo mode with predefined questions
    - Example question gallery
    - Knowledge base statistics
    - Beautiful formatted output
    - Error handling and recovery

Usage:
    # Run interactive demo
    python src/demo.py
    
    # Or import as module
    from src.demo import interactive_mode, demo_mode

Date: 2026
"""

import logging
import sys
from pathlib import Path
from typing import List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.vector_store import VectorStore, RAGRetriever
from src.query_engine import ISMSQueryEngine, ISMSAssistant

# Configure logger
logging.basicConfig(
    level=logging.WARNING,  # Suppress verbose logs in demo mode
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Demo questions in French and English
DEMO_QUESTIONS = {
    "fr": [
        {
            "question": "Quelles sont les exigences pour le contrôle d'accès physique ?",
            "framework": "iso27001",
        },
        {
            "question": "Comment gérer les incidents de sécurité ?",
            "framework": None,
        },
        {
            "question": "Quels sont les contrôles de gestion des actifs ?",
            "framework": "iso27001",
        },
    ],
    "en": [
        {
            "question": "What are the physical access control requirements?",
            "framework": "iso27001",
        },
        {
            "question": "How to manage security incidents?",
            "framework": None,
        },
        {
            "question": "What are the asset management controls?",
            "framework": "iso27001",
        },
    ],
}


def print_header(title: str, width: int = 70) -> None:
    """Print a formatted header."""
    print()
    print("=" * width)
    print(title.center(width))
    print("=" * width)
    print()


def print_separator(width: int = 70) -> None:
    """Print a separator line."""
    print("-" * width)


def show_example_questions() -> None:
    """Display example questions users can ask."""
    print_header("Example Questions")
    
    print("French Questions:")
    print()
    for i, item in enumerate(DEMO_QUESTIONS["fr"], 1):
        print(f"  {i}. {item['question']}")
        if item['framework']:
            print(f"     Framework: {item['framework']}")
    print()
    
    print("English Questions:")
    print()
    for i, item in enumerate(DEMO_QUESTIONS["en"], 1):
        print(f"  {i}. {item['question']}")
        if item['framework']:
            print(f"     Framework: {item['framework']}")
    print()
    
    print("Other Topics You Can Ask About:")
    print()
    print("  - Access control (contrôle d'accès)")
    print("  - Risk management (gestion des risques)")
    print("  - Incident response (réponse aux incidents)")
    print("  - Business continuity (continuité d'activité)")
    print("  - Cryptography (cryptographie)")
    print("  - Security policies (politiques de sécurité)")
    print("  - Compliance requirements (exigences de conformité)")
    print("  - Network security (sécurité réseau)")
    print()


def show_stats(vector_store: VectorStore) -> None:
    """Display knowledge base statistics."""
    print_header("Knowledge Base Statistics")
    
    try:
        stats = vector_store.get_collection_stats()
        
        print(f"Collection Name: {stats['collection_name']}")
        print(f"Total Documents: {stats['total_documents']}")
        print()
        
        if stats['frameworks']:
            print("Available Frameworks:")
            for fw in stats['frameworks']:
                print(f"  - {fw}")
            print()
        else:
            print("⚠ No frameworks loaded yet")
            print("  Run: python src/ingest_frameworks.py")
            print()
        
        if stats['total_documents'] > 0:
            print("Status: ✓ Ready for queries")
        else:
            print("Status: ⚠ Empty - please ingest documents")
        print()
        
    except Exception as e:
        print(f"✗ Error retrieving stats: {e}")
        print()


def format_answer(
    question: str,
    answer: str,
    sources: List,
    language: str = "en"
) -> None:
    """Format and display the answer with sources."""
    print_separator()
    print()
    if language == "fr":
        print(f"Question: {question}")
        print()
        print("Réponse:")
    else:
        print(f"Question: {question}")
        print()
        print("Answer:")
    print()
    print(answer)
    print()
    
    if sources:
        if language == "fr":
            print(f"Sources ({len(sources)} documents):")
        else:
            print(f"Sources ({len(sources)} documents):")
        print()
        
        for i, source in enumerate(sources[:3], 1):  # Show top 3 sources
            framework = source.metadata.get('framework', 'Unknown')
            filename = source.metadata.get('filename', 'Unknown')
            score = source.score
            print(f"  {i}. Framework: {framework}")
            print(f"     File: {filename}")
            print(f"     Relevance: {score:.2%}")
            print()
    print_separator()
    print()


def demo_mode(assistant: ISMSAssistant, language: str = "fr") -> None:
    """Run demo with predefined questions."""
    print_header(f"Demo Mode - {language.upper()}")
    
    questions = DEMO_QUESTIONS.get(language, DEMO_QUESTIONS["fr"])
    
    if language == "fr":
        print(f"Exécution de {len(questions)} questions de démonstration...")
    else:
        print(f"Running {len(questions)} demo questions...")
    print()
    
    for i, item in enumerate(questions, 1):
        question = item["question"]
        framework = item["framework"]
        
        print(f"[{i}/{len(questions)}] Processing: {question}")
        print()
        
        try:
            # Query with sources
            response = assistant.engine.query(
                question=question,
                framework=framework,
                language=language,
            )
            
            format_answer(question, response.answer, response.sources, language)
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            print(f"✗ Error: {e}")
            print()
    
    if language == "fr":
        print("✓ Démonstration terminée")
    else:
        print("✓ Demo completed")
    print()


def interactive_mode(assistant: ISMSAssistant) -> None:
    """Interactive question-answering mode."""
    print_header("Interactive Mode")
    
    print("Ask questions about security frameworks (type 'quit' to exit)")
    print("You can ask in French or English")
    print()
    print("Tips:")
    print("  - Type 'examples' to see example questions")
    print("  - Type 'stats' to see knowledge base statistics")
    print("  - Type 'lang fr' or 'lang en' to change language")
    print("  - Type 'help' for more commands")
    print()
    
    current_language = "fr"
    
    while True:
        print_separator()
        try:
            # Get question from user
            question = input("\n> ").strip()
            
            if not question:
                continue
            
            # Handle commands
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            elif question.lower() == 'examples':
                show_example_questions()
                continue
            
            elif question.lower() == 'stats':
                show_stats(assistant.engine.retriever.vector_store)
                continue
            
            elif question.lower().startswith('lang '):
                lang = question.lower().split()[1]
                if lang in ['fr', 'en']:
                    current_language = lang
                    print(f"\n✓ Language changed to: {lang}")
                else:
                    print("\n✗ Invalid language. Use 'fr' or 'en'")
                continue
            
            elif question.lower() == 'help':
                print("\nAvailable Commands:")
                print("  examples - Show example questions")
                print("  stats    - Show knowledge base statistics")
                print("  lang fr  - Switch to French")
                print("  lang en  - Switch to English")
                print("  help     - Show this help")
                print("  quit     - Exit interactive mode")
                print()
                continue
            
            # Process question
            print()
            print("Searching knowledge base...")
            
            try:
                response = assistant.engine.query(
                    question=question,
                    language=current_language,
                )
                
                print()
                format_answer(question, response.answer, response.sources, current_language)
                
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                print(f"\n✗ Error: {e}")
                print("Please try rephrasing your question or check the knowledge base.\n")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit.")
            continue
        except EOFError:
            print("\n\nGoodbye!")
            break


def main() -> int:
    """
    Main demo entry point with menu system.
    
    Returns:
        Exit code (0 = success, 1 = failure)
    """
    print_header("ISMS RAG Platform - Interactive Demo")
    
    print("Initializing platform...")
    print()
    
    # Initialize components
    try:
        vector_store = VectorStore(
            collection_name=Config.CHROMA_COLLECTION_NAME,
            persist_directory=Config.CHROMA_PERSIST_DIRECTORY,
        )
        
        retriever = RAGRetriever(vector_store)
        
        # Use Ollama by default (free local LLM)
        engine = ISMSQueryEngine(
            retriever=retriever,
            language="fr",
            model="llama3.2",
            use_ollama=True,
        )
        
        assistant = ISMSAssistant(engine)
        
        print("✓ Platform initialized successfully")
        print(f"  - Using model: llama3.2 (Ollama)")
        print(f"  - Embedding model: {Config.EMBEDDING_MODEL}")
        print()
        
    except Exception as e:
        logger.error(f"Failed to initialize platform: {e}")
        print(f"✗ Error: {e}")
        print()
        print("Please ensure:")
        print("  1. Ollama is installed and running")
        print("  2. Vector store is initialized")
        print("  3. Run: python src/ingest_frameworks.py")
        return 1
    
    # Check if knowledge base has data
    stats = vector_store.get_collection_stats()
    if stats['total_documents'] == 0:
        print("⚠ Warning: Knowledge base is empty")
        print("  Please run: python src/ingest_frameworks.py")
        print()
        response = input("Continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            return 0
        print()
    
    # Main menu
    while True:
        print_header("Main Menu")
        
        print("Choose an option:")
        print()
        print("  1. Interactive Mode - Ask your own questions")
        print("  2. Demo Mode (French) - Run French demo questions")
        print("  3. Demo Mode (English) - Run English demo questions")
        print("  4. Show Example Questions")
        print("  5. Show Knowledge Base Statistics")
        print("  6. Quit")
        print()
        
        try:
            choice = input("Enter choice (1-6): ").strip()
            
            if choice == '1':
                interactive_mode(assistant)
            
            elif choice == '2':
                demo_mode(assistant, language="fr")
            
            elif choice == '3':
                demo_mode(assistant, language="en")
            
            elif choice == '4':
                show_example_questions()
            
            elif choice == '5':
                show_stats(vector_store)
            
            elif choice == '6':
                print()
                print("Thank you for using ISMS RAG Platform!")
                print()
                return 0
            
            else:
                print()
                print("✗ Invalid choice. Please enter 1-6.")
                print()
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            return 0
        except EOFError:
            print("\n\nGoodbye!")
            return 0


if __name__ == "__main__":
    sys.exit(main())

