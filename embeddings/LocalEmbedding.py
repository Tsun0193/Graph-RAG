import os
import logging
import sys
import nest_asyncio
import json
import torch

import numpy as np
from typing import List, Any
from llama_index.core.embeddings import BaseEmbedding

from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.core import (
    Settings,
    Document, get_response_synthesizer,
    KnowledgeGraphIndex, PropertyGraphIndex,
    StorageContext
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.graph_stores.neo4j import Neo4jGraphStore

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True, model_kwargs={"torch_dtype":"bfloat16"})

def get_embeddings(text: str,
                   model: str,
                   **kwargs) -> List:
    """
    Get embeddings for a given text using a local pre-trained model.
    """
    # TODO: Implement this function
    _model = SentenceTransformer(model, trust_remote_code=True, model_kwargs={"torch_dtype":"bfloat16"})
    embeddings = _model.encode(text, batch_size=32, show_progress_bar=True)
    embeddings = embeddings.tolist()
    return embeddings

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
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            embeddings.append(self._get_text_embedding(text))
        return embeddings
