import numpy as np
from typing import List, Any
from llama_index.core.embeddings import BaseEmbedding

def get_embeddings(text: str,
                   model: str,
                   **kwargs) -> np.ndarray:
    """
    Get embeddings for a given text using a local pre-trained model.
    """
    # TODO: Implement this function
    pass

class LocalEmbedding(BaseEmbedding):
    """
    Local Embedding class.
    """
    def __init__(self,
                 model: str,
                 **kwargs: Any) -> None:
        """
        Initialize the LocalEmbedding class.
        """
        super().__init__(**kwargs)
        self._model = model

    @classmethod
    def class_name(cls) -> str:
        """
        Get the class name.
        """
        return "LocalEmbedding"
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """
        Get the query embedding.
        """
        return self._get_query_embedding(query)
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        """
        Get the text embedding.
        """
        return self._get_text_embedding(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        embeddings = get_embeddings(query, self._model)
        return embeddings.tolist()
    
    def _get_text_embedding(self, text: str) -> List[float]:
        embeddings = get_embeddings(text, self._model)
        return embeddings.tolist()
    
    def _get_text_embeddings(self, texts):
        embeddings = []
        for text in texts:
            embeddings.append(self._get_text_embedding(text))
        return embeddings
