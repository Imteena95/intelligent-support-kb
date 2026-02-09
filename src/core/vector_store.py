import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import os
import logging
from uuid import uuid4

logger = logging.getLogger(__name__)

class FreeVectorStore:
    def __init__(self, persist_dir: str = "./chroma_db", collection_name: str = "support_kb"):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Customer support knowledge base"}
        )
        logger.info(f"Vector store initialized: {persist_dir}/{collection_name}")
    
    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict]] = None, ids: Optional[List[str]] = None):
        if not documents:
            logger.warning("No documents to add")
            return []
        if ids is None:
            ids = [str(uuid4()) for _ in range(len(documents))]
        self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
        logger.info(f"Added {len(documents)} documents to vector store")
        return ids
    
    def search(self, query_embedding: List[float], n_results: int = 5) -> List[Dict[str, Any]]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                formatted_results.append({
                    'document': doc,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'id': results['ids'][0][i],
                    'distance': results['distances'][0][i] if results['distances'] else None
                })
        return formatted_results
    
    def get_stats(self) -> Dict[str, Any]:
        count = self.collection.count()
        return {
            "total_documents": count,
            "collection": self.collection_name,
            "location": self.persist_dir
        }
    
    def clear(self):
        self.collection.delete(where={})
        logger.info("Vector store cleared")

_vector_store = None

def get_vector_store(persist_dir: str = "./chroma_db", collection_name: str = "support_kb") -> FreeVectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = FreeVectorStore(persist_dir, collection_name)
    return _vector_store