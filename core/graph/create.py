import os
import logging
import sys
import nest_asyncio
import json

sys.path.append('/home/tsunn/Workspace/iai-lab/sosci/codes/Graph-RAG')


from argparse import ArgumentParser
from dotenv import load_dotenv
from typing import List
from llama_index.core import (
    Settings,
    Document, get_response_synthesizer,
    KnowledgeGraphIndex, PropertyGraphIndex,
    StorageContext
)
from llama_index.graph_stores.neo4j import Neo4jGraphStore


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("llama_index").setLevel(logging.INFO)
nest_asyncio.apply()
load_dotenv()

CREATE_QUERY = """
MERGE (chunk: Chunk {id: $id})
ON CREATE SET 
    chunk.length = $length,
    chunk.fufll_text = $full_text
WITH chunk
UNWIND $keywords AS keyword
MERGE (k: Keyword {name: keyword})
MERGE (chunk)-[:HAS_KEYWORD]->(k)
RETURN chunk
"""

def instantiate(
    chunks: List[Document] = None,
    username: str = os.environ["NEO4J_USERNAME"],
    password: str = os.environ["NEO4J_PASSWORD"],
    url: str = os.environ["NEO4J_URI"],
    embedding_model_name: str = None
) -> None:
    assert chunks, "Chunks must be provided"
    assert username, "Username must be provided"
    assert password, "Password must be provided"
    assert url, "URL must be provided"

    graph_store = Neo4jGraphStore(
        username=username,
        password=password,
        url=url
    )
    print(f"Configurations loaded with username: {username} at url: {url}")

    for chunk in chunks:
        graph_store.query(CREATE_QUERY, {
            "id": chunk.metadata["id"],
            "length": chunk.metadata["length"],
            "full_text": chunk.text,
            "keywords": chunk.metadata["keywords"]
        })
    print(f"Finish indexing {len(chunks)} nodes")

    graph_store.query(
        """
        CREATE CONSTRAINT unique_chunk IF NOT EXISTS
        FOR (chunk:Chunk) REQUIRE c.id IS UNIQUE
        """
    )
    print("Unique constraint created")


if __name__ == "__main__":
    print(os.getcwd())
    parser = ArgumentParser()
    parser.add_argument("--datapath", type=str, required=True)
    parser.add_argument("--embeddings", type=bool, default=True)
    args = parser.parse_args()

    instantiate(datapath=args.datapath, 
                include_embeddings=args.embeddings, 
                max_triplets_per_chunk=args.max_triplets_per_chunk)
    print("Indexing complete!")