from typing import List, Any
from llama_index.core.embeddings import BaseEmbedding

from sentence_transformers import SentenceTransformer

# model = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True, model_kwargs={"torch_dtype":"bfloat16"})

# def get_embeddings(text: str,
#                    model: SentenceTransformer,
#                    **kwargs) -> List:
#     """
#     Get embeddings for a given text using a local pre-trained model.
#     """
#     embeddings = model.encode(text, **kwargs)
#     embeddings = embeddings.tolist()
#     return embeddings


class LocalEmbedding(BaseEmbedding):
    """
    Local Embedding class.
    """
    def __init__(self,
                 model: str = "Alibaba-NLP/gte-large-en-v1.5",
                 **kwargs: Any) -> None:
        """
        Initialize the LocalEmbedding class.
        """
        super().__init__(**kwargs)
        self._model = SentenceTransformer(model, trust_remote_code=True, model_kwargs={"torch_dtype":"bfloat16"})
        print(f"Loaded model: {model}")

    @classmethod
    def class_name(cls) -> str:
        """
        Get the class name.
        """
        return "LocalEmbedding"

    def get_embeddings(self,
                       text: str,
                       **kwargs) -> List:
        """
        Get embeddings for a given text using a local pre-trained model.
        """
        embeddings = self._model.encode(text, **kwargs)
        embeddings = embeddings.tolist()
        return embeddings
    
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
        return self.get_embeddings(query)
    
    def _get_text_embedding(self, text: str) -> List[float]:
        return self.get_embeddings(text)
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [self.get_embeddings(text) for text in texts]
