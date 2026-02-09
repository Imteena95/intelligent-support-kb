import logging
from typing import List, Dict, Any
from .embeddings import get_embeddings
from .vector_store import get_vector_store
from .llm import get_llm
import os

logger = logging.getLogger(__name__)

class FreeRAGPipeline:
    def __init__(self):
        embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        llm_model = os.getenv("LOCAL_MODEL", "llama2:7b")
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        collection = os.getenv("CHROMA_COLLECTION", "support_kb")
        self.embeddings = get_embeddings(embedding_model)
        self.vector_store = get_vector_store(persist_dir, collection)
        self.llm = get_llm(llm_model)
        logger.info("FREE RAG Pipeline initialized")
    
    def ingest_documents(self, documents: List[str], metadatas=None):
        if not documents:
            return {"error": "No documents provided"}
        logger.info(f"Ingesting {len(documents)} documents...")
        ids = self.vector_store.add_documents(documents, metadatas)
        return {"status": "success", "document_count": len(documents), "ids": ids}
    
    def query(self, question: str, n_results: int = 3) -> Dict[str, Any]:
        logger.info(f"Querying: {question}")
        query_embedding = self.embeddings.embed_query(question)
        search_results = self.vector_store.search(query_embedding, n_results)
        if not search_results:
            return {
                "question": question,
                "context": [],
                "answer": "No relevant information found.",
                "sources": []
            }
        context_docs = [result['document'] for result in search_results]
        context = "\n\n".join(context_docs)
        answer = self.llm.generate_with_context(question, context)
        sources = [
            {
                "content": result['document'][:200] + "...",
                "metadata": result['metadata'],
                "id": result['id']
            }
            for result in search_results
        ]
        return {
            "question": question,
            "context": context_docs,
            "answer": answer,
            "sources": sources,
            "search_count": len(search_results)
        }
    
    def get_stats(self) -> Dict[str, Any]:
        vector_stats = self.vector_store.get_stats()
        return {
            "vector_store": vector_stats,
            "embedding_model": self.embeddings.get_dimension(),
            "llm_model": self.llm.model_name
        }

rag_pipeline = None

def get_rag_pipeline() -> FreeRAGPipeline:
    global rag_pipeline
    if rag_pipeline is None:
        rag_pipeline = FreeRAGPipeline()
    return rag_pipeline