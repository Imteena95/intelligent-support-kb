"""
Simple test for RAG system
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_imports():
    """Test that we can import all modules"""
    try:
        from src.core.embeddings import FreeEmbeddings
        from src.core.vector_store import FreeVectorStore
        from src.core.llm import FreeLLM
        from src.core.rag_pipeline import FreeRAGPipeline
        print("✅ All imports successful!")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing FREE RAG System...")
    test_imports()