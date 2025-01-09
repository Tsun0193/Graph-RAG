import os
import logging
import sys
import nest_asyncio

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


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("llama_index").setLevel(logging.INFO)
nest_asyncio.apply()
load_dotenv()

if __name__ == "__main__":
    graph_store = Neo4jGraphStore(
        username=os.environ["NEO4J_USERNAME"],
        password=os.environ["NEO4J_PASSWORD"],
        url=os.environ["NEO4J_URI"]
    )

    print("Deleting all nodes in the graph")

    graph_store.query(
    """
    MATCH (n) DETACH DELETE n
    """
    )

    print("All nodes deleted!")