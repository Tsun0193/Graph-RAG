import os
import logging
import sys
import nest_asyncio
import json
import re

from dotenv import load_dotenv
from typing import List, Literal, Dict
from tqdm.auto import tqdm
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document
from llama_index.core.embeddings import BaseEmbedding
from embeddings.LocalEmbedding import LocalEmbedding


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("llama_index").setLevel(logging.INFO)
nest_asyncio.apply()
load_dotenv()

embedder = LocalEmbedding()
# BATCH_SIZE = 16

def vectorize(chunks: List[Dict],
              embed_model: BaseEmbedding = embedder,
            #   batch_size: int = BATCH_SIZE,
              show_progress: bool = True,
              **kwargs) -> List[List[float]]:
    """
    Vectorize a list of texts using the provided embedding model.
    """
    embeddings = []
    texts = [chunk["text"] for chunk in chunks]

    for text in tqdm(texts, desc="Vectorizing", total=len(texts), disable=not show_progress):
        embeddings.append(embed_model._get_text_embedding(text))

    return embeddings

if __name__ == "__main__":
    pass