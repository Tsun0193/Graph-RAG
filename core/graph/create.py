import os
import logging
import sys
import nest_asyncio
import json

sys.path.append('/home/tsunn/Workspace/iai-lab/sosci/codes/Graph-RAG')


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
from embeddings.LocalEmbedding import LocalEmbedding
from llm.TogetherLLM import TogetherLLM


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("llama_index").setLevel(logging.INFO)
nest_asyncio.apply()
load_dotenv()


def instantiate(
    datapath: str,
    include_embeddings: bool = False,
    max_triplets_per_chunk: int = 3,
    username: str = os.environ["NEO4J_USERNAME"],
    password: str = os.environ["NEO4J_PASSWORD"],
    url: str = os.environ["NEO4J_URI"],
    embedding_model_name: str = None,
    llm_model_name: str = None,
) -> None:
    try:
        chunks = process_data(datapath)
    except Exception as e:
        print(f"Error processing data: {e}")
        sys.exit(1)

    chunks = chunks[:10]
    print(len(chunks))

    if embedding_model_name:
        embedder = LocalEmbedding(model=embedding_model_name)
    else:
        embedder = LocalEmbedding()

    if llm_model_name:
        llm = TogetherLLM(model=llm_model_name)
    else:
        llm = TogetherLLM()

    graph_store = Neo4jGraphStore(
        username=username,
        password=password,
        url=url
    )
    print("Configurations loaded with username: {username} at url: {url}")

    storage_context = StorageContext.from_defaults(graph_store=graph_store)

    index = KnowledgeGraphIndex(
        nodes=chunks,
        llm=llm,
        embed_model=embedder,
        storage_context=storage_context,
        include_embeddings=include_embeddings,
        max_triplets_per_chunk=max_triplets_per_chunk,
        show_progress=True
    )

    print(f"Finish indexing {len(chunks)} nodes")


if __name__ == "__main__":
    print(os.getcwd())
    parser = ArgumentParser()
    parser.add_argument("--datapath", type=str, required=True)
    parser.add_argument("--embeddings", type=bool, default=False)
    parser.add_argument("--max_triplets_per_chunk", type=int, default=2)
    args = parser.parse_args()

    instantiate(args.datapath, args.embeddings, args.max_triplets_per_chunk)
    print("Indexing complete!")