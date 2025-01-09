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


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("llama_index").setLevel(logging.INFO)
nest_asyncio.apply()
load_dotenv()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--datapath", type=str, required=True)
    parser.add_argument("--embeddings", type=bool, default=False)
    args = parser.parse_args()

    assert os.path.exists(args.datapath), "Data path does not exist"
    file_type = args.datapath.split(".")[-1]
    splitter = SentenceSplitter(separator=".")

    if file_type == "txt":
        with open(args.datapath, 'r') as f:
            text = f.read()
        data = Document(text=text)
        chunks = splitter.get_nodes_from_documents([data], show_progress=True)
        print(f"Finish processing {len(chunks)} chunks")
        
    elif file_type == "jsonl":
        data = []
        with open(args.datapath, 'r') as f:
            for line in f:
                line = json.loads(line)
                for doc in line['documents']:
                    data.append(Document(text=doc['text']))
        chunks = splitter.get_nodes_from_documents(data, show_progress=True)
        print(f"Finish processing {len(chunks)} chunks")

    graph_store = Neo4jGraphStore(
        username=os.environ["NEO4J_USERNAME"],
        password=os.environ["NEO4J_PASSWORD"],
        url=os.environ["NEO4J_URI"]
    )

    storage_context = StorageContext.from_defaults(graph_store=graph_store)

    index = KnowledgeGraphIndex(
        nodes=chunks,
        storage_context=storage_context,
        include_embeddings=args.embeddings,
        show_progress=True
    )

    print(f"Finish indexing {len(chunks)} nodes")