import logging
from typing import List
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class FreeEmbeddings:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        logger.info(f"Loading FREE embeddings: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embeddings loaded. Dimension: {self.dimension}")
    
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        if not documents:
            return []
        embeddings = self.model.encode(documents, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.tolist()
    
    def embed_query(self, query: str) -> List[float]:
        embedding = self.model.encode([query], convert_to_numpy=True)
        return embedding[0].tolist()
    
    def get_dimension(self) -> int:
        return self.dimension

_embeddings = None

def get_embeddings(model_name: str = "all-MiniLM-L6-v2") -> FreeEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = FreeEmbeddings(model_name)
    return _embeddings