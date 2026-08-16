"""
RAG Query Engine for ISMS Platform
==================================

This module provides RAG (Retrieval Augmented Generation) capabilities for the ISMS platform
with bilingual support (French/English) and framework-specific query handling.

The module includes:
- ISMSQueryEngine: Core RAG engine with bilingual prompts
- ISMSAssistant: High-level conversational interface
- Prompt templates optimized for security frameworks
- Source citation and metadata tracking

Features:
    - Bilingual prompt templates (French/English)
    - Context retrieval from vector store
    - OpenAI GPT integration
    - Source citation with metadata
    - Framework-specific responses
    - Control explanation capabilities

Usage:
    from src.config import Config
    from src.vector_store import VectorStore, RAGRetriever
    from src.query_engine import ISMSQueryEngine, ISMSAssistant
    
    # Initialize components
    vector_store = VectorStore()
    retriever = RAGRetriever(vector_store)
    engine = ISMSQueryEngine(retriever=retriever, language="fr")
    
    # Query
    response = engine.query("Quelles sont les exigences pour le contrôle d'accès ?")
    
    # Use assistant
    assistant = ISMSAssistant(engine)
    answer = assistant.ask("Access control requirements", framework="iso27001")

Date: 2026
"""

import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
import requests
import json

from openai import OpenAI

from src.config import Config
from src.vector_store import VectorStore, RAGRetriever, SearchResult

# Configure logger
logger = logging.getLogger(__name__)


@dataclass
class QueryResponse:
    """
    Container for query responses.
    
    Attributes:
        answer: Generated answer text
        sources: List of source documents used
        language: Language of the response
        framework: Framework filter used (if any)
        context_used: Number of context chunks used
    """
    answer: str
    sources: List[SearchResult]
    language: str
    framework: Optional[str] = None
    context_used: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary format."""
        return {
            "answer": self.answer,
            "sources": [
                {
                    "content": s.content[:200] + "..." if len(s.content) > 200 else s.content,
                    "framework": s.metadata.get("framework"),
                    "filename": s.metadata.get("filename"),
                    "score": s.score,
                }
                for s in self.sources
            ],
            "language": self.language,
            "framework": self.framework,
            "context_used": self.context_used,
        }


class ISMSQueryEngine:
    """
    RAG query engine for ISMS platform with bilingual support.
    
    This class handles RAG queries with context retrieval from the vector store
    and generates responses using OpenAI GPT models.
    
    Attributes:
        retriever: RAGRetriever instance for context retrieval
        language: Default language ("fr" or "en")
        model: OpenAI model name
        temperature: Model temperature
        max_tokens: Maximum tokens in response
        client: OpenAI client instance
    """
    
    # French prompt template
    PROMPT_TEMPLATE_FR = """Tu es un expert en sécurité de l'information et systèmes de management de la sécurité de l'information (SMSI).
Ta mission est de fournir des réponses précises et détaillées basées sur les normes et frameworks de sécurité.

Contexte pertinent extrait des documents:
{context}

Question: {question}

Instructions:
- Réponds en français de manière claire et professionnelle
- Base ta réponse sur le contexte fourni
- Cite les contrôles spécifiques (ex: A.5.1, A.6.1) quand c'est pertinent
- Si le contexte ne contient pas assez d'informations, indique-le clairement
- Structure ta réponse avec des points ou sections si nécessaire
- Mentionne le framework concerné (ISO 27001, NIST CSF, CIS Controls, etc.)

Réponse:"""

    # English prompt template
    PROMPT_TEMPLATE_EN = """You are an expert in information security and Information Security Management Systems (ISMS).
Your mission is to provide accurate and detailed answers based on security standards and frameworks.

Relevant context extracted from documents:
{context}

Question: {question}

Instructions:
- Answer in English clearly and professionally
- Base your answer on the provided context
- Cite specific controls (e.g., A.5.1, A.6.1) when relevant
- If the context doesn't contain enough information, clearly state it
- Structure your answer with bullet points or sections if necessary
- Mention the relevant framework (ISO 27001, NIST CSF, CIS Controls, etc.)

Answer:"""

    def __init__(
        self,
        retriever: RAGRetriever,
        language: str = "fr",
        model: str = Config.LLM_MODEL,
        temperature: float = Config.LLM_TEMPERATURE,
        max_tokens: int = Config.MAX_TOKENS,
        use_ollama: bool = True,  # Default to Ollama
        ollama_url: str = "http://localhost:11434",
    ):
        """
        Initialize ISMSQueryEngine.
        
        Args:
            retriever: RAGRetriever instance
            language: Default language ("fr" or "en")
            model: Model name (OpenAI or Ollama)
            temperature: Model temperature for randomness
            max_tokens: Maximum tokens in response
            use_ollama: If True, use Ollama; if False, use OpenAI
            ollama_url: Ollama API endpoint
        """
        self.retriever = retriever
        self.language = language.lower()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_ollama = use_ollama
        self.ollama_url = ollama_url
        
        # Initialize LLM client
        if not self.use_ollama:
            self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
            logger.info(f"ISMSQueryEngine initialized with OpenAI (language={language}, model={model})")
        else:
            logger.info(f"ISMSQueryEngine initialized with Ollama (language={language}, model={model})")
    
    def _get_prompt_template(self, language: Optional[str] = None) -> str:
        """
        Get prompt template for specified language.
        
        Args:
            language: Language code ("fr" or "en"), uses default if None
        
        Returns:
            Prompt template string
        """
        lang = (language or self.language).lower()
        
        if lang == "fr":
            return self.PROMPT_TEMPLATE_FR
        else:
            return self.PROMPT_TEMPLATE_EN
    
    def _build_context(self, results: List[SearchResult]) -> str:
        """
        Build context string from search results.
        
        Args:
            results: List of SearchResult objects
        
        Returns:
            Formatted context string with sources
        """
        if not results:
            return "Aucun contexte pertinent trouvé." if self.language == "fr" else "No relevant context found."
        
        context_parts = []
        
        for i, result in enumerate(results, 1):
            source = result.metadata.get("filename", "unknown")
            framework = result.metadata.get("framework", "unknown")
            
            context_parts.append(
                f"[Source {i}: {source} ({framework})]\n{result.content}"
            )
        
        return "\n\n---\n\n".join(context_parts)
    
    def query(
        self,
        question: str,
        framework: Optional[str] = None,
        language: Optional[str] = None,
        k: int = Config.TOP_K_RESULTS,
        min_score: float = Config.SIMILARITY_THRESHOLD,
    ) -> QueryResponse:
        """
        Execute RAG query.
        
        Args:
            question: User question
            framework: Optional framework filter
            language: Optional language override
            k: Number of context chunks to retrieve
            min_score: Minimum similarity score for retrieval
        
        Returns:
            QueryResponse with answer and sources
        
        Raises:
            Exception: If query processing fails
        """
        logger.info(f"Query: '{question}' (framework={framework}, k={k})")
        
        # Retrieve relevant context
        results = self.retriever.retrieve(
            query=question,
            k=k,
            framework=framework,
            min_score=min_score,
        )
        
        logger.info(f"Retrieved {len(results)} relevant documents")
        
        # Build context
        context = self._build_context(results)
        
        # Get prompt template
        prompt_template = self._get_prompt_template(language)
        
        # Format prompt
        prompt = prompt_template.format(
            context=context,
            question=question,
        )
        
        # Generate response
        try:
            if self.use_ollama:
                # Use Ollama
                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": f"You are an expert in information security management systems.\n\n{prompt}",
                        "stream": False,
                        "options": {
                            "temperature": self.temperature,
                            "num_predict": self.max_tokens,
                        }
                    },
                    timeout=120
                )
                response.raise_for_status()
                answer = response.json()["response"].strip()
            else:
                # Use OpenAI
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert in information security management systems."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                answer = response.choices[0].message.content.strip()
            
            logger.info(f"Generated answer ({len(answer)} chars)")
            
            return QueryResponse(
                answer=answer,
                sources=results,
                language=language or self.language,
                framework=framework,
                context_used=len(results),
            )
        
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise


class ISMSAssistant:
    """
    High-level conversational interface for ISMS queries.
    
    This class provides convenient methods for common ISMS query patterns
    and formats responses in a user-friendly way.
    
    Attributes:
        engine: ISMSQueryEngine instance
    """
    
    def __init__(self, engine: ISMSQueryEngine):
        """
        Initialize ISMSAssistant.
        
        Args:
            engine: ISMSQueryEngine instance
        """
        self.engine = engine
        logger.info("ISMSAssistant initialized")
    
    def ask(
        self,
        question: str,
        framework: Optional[str] = None,
        language: Optional[str] = None,
    ) -> str:
        """
        Ask a question and get a simple text answer.
        
        Args:
            question: User question
            framework: Optional framework filter
            language: Optional language override
        
        Returns:
            Answer text
        """
        response = self.engine.query(
            question=question,
            framework=framework,
            language=language,
        )
        
        return response.answer
    
    def ask_with_sources(
        self,
        question: str,
        framework: Optional[str] = None,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Ask a question and get answer with source citations.
        
        Args:
            question: User question
            framework: Optional framework filter
            language: Optional language override
        
        Returns:
            Dictionary with answer and sources
        """
        response = self.engine.query(
            question=question,
            framework=framework,
            language=language,
        )
        
        return response.to_dict()
    
    def explain_control(
        self,
        control_id: str,
        framework: str = "iso27001",
        language: Optional[str] = None,
    ) -> str:
        """
        Explain a specific security control.
        
        Args:
            control_id: Control identifier (e.g., "A.5.1", "AC-1")
            framework: Framework name
            language: Optional language override
        
        Returns:
            Control explanation
        """
        lang = language or self.engine.language
        
        if lang == "fr":
            question = f"Explique le contrôle {control_id} en détail."
        else:
            question = f"Explain control {control_id} in detail."
        
        response = self.engine.query(
            question=question,
            framework=framework,
            language=language,
        )
        
        return response.answer
    
    def compare_frameworks(
        self,
        topic: str,
        frameworks: List[str],
        language: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Compare how different frameworks address a topic.
        
        Args:
            topic: Topic to compare
            frameworks: List of framework names
            language: Optional language override
        
        Returns:
            Dictionary mapping framework names to answers
        """
        lang = language or self.engine.language
        
        if lang == "fr":
            question = f"Comment {topic} est-il traité dans ce framework ?"
        else:
            question = f"How is {topic} addressed in this framework?"
        
        results = {}
        
        for framework in frameworks:
            response = self.engine.query(
                question=question,
                framework=framework,
                language=language,
            )
            results[framework] = response.answer
        
        return results
    
    def get_recommendations(
        self,
        scenario: str,
        framework: Optional[str] = None,
        language: Optional[str] = None,
    ) -> str:
        """
        Get security recommendations for a scenario.
        
        Args:
            scenario: Security scenario description
            framework: Optional framework filter
            language: Optional language override
        
        Returns:
            Recommendations text
        """
        lang = language or self.engine.language
        
        if lang == "fr":
            question = f"Quelles sont les recommandations de sécurité pour le scénario suivant : {scenario}"
        else:
            question = f"What are the security recommendations for the following scenario: {scenario}"
        
        response = self.engine.query(
            question=question,
            framework=framework,
            language=language,
        )
        
        return response.answer


def test_query_engine():
    """
    Test function for ISMSQueryEngine and ISMSAssistant.
    
    Tests:
    - Query engine initialization
    - Basic queries in French and English
    - Assistant interface
    - Control explanation
    - Source retrieval
    """
    print("=" * 70)
    print("Testing ISMSQueryEngine")
    print("=" * 70)
    print()
    
    # Test initialization
    print("1. Initializing components...")
    try:
        from src.document_processor import DocumentProcessor
        
        # Create and populate vector store
        vector_store = VectorStore(
            collection_name="test_query_engine",
            persist_directory="./test_query_chroma_db",
        )
        
        # Add test documents
        test_file = "data/frameworks/tests/test_iso27001.txt"
        processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
        chunks = processor.process_document(test_file)
        vector_store.add_documents(chunks, show_progress=False)
        
        # Initialize retriever and engine
        retriever = RAGRetriever(vector_store)
        engine = ISMSQueryEngine(retriever=retriever, language="fr")
        
        print(f"   ✓ Components initialized")
        print(f"   - Documents: {len(chunks)}")
        print(f"   - Language: {engine.language}")
        print(f"   - Model: {engine.model}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return
    print()
    
    # Test French query
    print("2. Testing French query...")
    try:
        question_fr = "Quels sont les contrôles de sécurité physique ?"
        response = engine.query(question_fr, framework="iso27001")
        
        print(f"   ✓ Query processed")
        print(f"   - Question: {question_fr}")
        print(f"   - Sources used: {response.context_used}")
        print(f"   - Answer length: {len(response.answer)} chars")
        print(f"\n   Answer preview:")
        print(f"   {response.answer[:200]}...")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
    print()
    
    # Test English query
    print("3. Testing English query...")
    try:
        question_en = "What are the access control requirements?"
        response = engine.query(question_en, language="en", framework="iso27001")
        
        print(f"   ✓ Query processed")
        print(f"   - Question: {question_en}")
        print(f"   - Sources used: {response.context_used}")
        print(f"   - Answer length: {len(response.answer)} chars")
        print(f"\n   Answer preview:")
        print(f"   {response.answer[:200]}...")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    print()
    
    # Test assistant
    print("4. Testing ISMSAssistant...")
    try:
        assistant = ISMSAssistant(engine)
        
        # Simple ask
        answer = assistant.ask("C'est quoi la gestion des actifs ?")
        print(f"   ✓ Simple ask: {len(answer)} chars")
        
        # Ask with sources
        result = assistant.ask_with_sources("Risk assessment process")
        print(f"   ✓ Ask with sources: {len(result['sources'])} sources")
        
        # Explain control
        explanation = assistant.explain_control("A.7.1", framework="iso27001")
        print(f"   ✓ Control explanation: {len(explanation)} chars")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
    print()
    
    # Test source metadata
    print("5. Testing source metadata...")
    try:
        response = engine.query("Compliance requirements", framework="iso27001")
        
        if response.sources:
            print(f"   ✓ Retrieved {len(response.sources)} sources")
            for i, source in enumerate(response.sources[:2], 1):
                print(f"\n   Source {i}:")
                print(f"   - Framework: {source.metadata.get('framework')}")
                print(f"   - Filename: {source.metadata.get('filename')}")
                print(f"   - Score: {source.score:.3f}")
                print(f"   - Preview: {source.content[:80]}...")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    print()
    
    # Cleanup
    print("6. Cleaning up...")
    try:
        vector_store.delete_collection()
        print(f"   ✓ Collection deleted")
        
        import shutil
        from pathlib import Path
        test_dir = Path("./test_query_chroma_db")
        if test_dir.exists():
            shutil.rmtree(test_dir, ignore_errors=True)
            print(f"   ✓ Test directory removed")
    except Exception as e:
        print(f"   ⚠ Cleanup warning: {e}")
    
    print()
    print("=" * 70)
    print("Query Engine test complete")
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
    
    test_query_engine()

