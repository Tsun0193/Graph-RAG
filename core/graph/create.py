import os
import logging
import sys
import nest_asyncio
import json

from argparse import ArgumentParser
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
from core.data.processing import process_data


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("llama_index").setLevel(logging.INFO)
nest_asyncio.apply()
load_dotenv()


def instantiate(
    datapath: str,
    include_embeddings: bool = False,
    max_triplets_per_chunk: int = 2,
    username: str = os.environ["NEO4J_USERNAME"],
    password: str = os.environ["NEO4J_PASSWORD"],
    url: str = os.environ["NEO4J_URI"]
) -> None:
    try:
        chunks = process_data(datapath)
    except Exception as e:
        print(f"Error processing data: {e}")
        sys.exit(1)

    graph_store = Neo4jGraphStore(
        username=username,
        password=password,
        url=url
    )
    print("Configurations loaded with username: {username} at url: {url}")

    storage_context = StorageContext.from_defaults(graph_store=graph_store)

    index = KnowledgeGraphIndex(
        nodes=chunks,
        storage_context=storage_context,
        include_embeddings=include_embeddings,
        max_triplets_per_chunk=max_triplets_per_chunk,
        show_progress=True
    )

    print(f"Finish indexing {len(chunks)} nodes")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--datapath", type=str, required=True)
    parser.add_argument("--embeddings", type=bool, default=False)
    parser.add_argument("--max_triplets_per_chunk", type=int, default=2)
    args = parser.parse_args()

    instantiate(args.datapath, args.embeddings, args.max_triplets_per_chunk)
    print("Indexing complete!")