import os
import logging
import sys
import nest_asyncio
import json

from dotenv import load_dotenv
from typing import List
from llama_index.llms.openai import OpenAI
from llama_index.core import (
    Settings,
    Document, get_response_synthesizer,
    KnowledgeGraphIndex, PropertyGraphIndex,
    StorageContext
)
from llama_index.core.node_parser import SentenceSplitter


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("llama_index").setLevel(logging.INFO)
nest_asyncio.apply()
load_dotenv()

def process_data(datapath: str) -> List[Document]:
    assert os.path.exists(datapath), "Data path does not exist"
    file_type = datapath.split(".")[-1]
    splitter = SentenceSplitter(separator=".")

    if file_type == "txt":
        with open(datapath, 'r') as f:
            text = f.read()
        data = Document(text=text)
        chunks = splitter.get_nodes_from_documents([data], show_progress=True)
        print(f"Finish processing {len(chunks)} chunks")
        
    elif file_type == "jsonl":
        data = []
        with open(datapath, 'r') as f:
            for line in f:
                line = json.loads(line)
                for doc in line['documents']:
                    data.append(Document(text=doc['text']))
        chunks = splitter.get_nodes_from_documents(data, show_progress=True)
        print(f"Finish processing {len(chunks)} chunks")

    return chunks

if __name__ == "__main__":
    pass