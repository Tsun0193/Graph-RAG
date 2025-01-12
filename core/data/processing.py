import os
import logging
import sys
import nest_asyncio
import json
import re

from dotenv import load_dotenv
from typing import List, Literal, Dict
from tqdm.auto import tqdm
from collections import Counter
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("llama_index").setLevel(logging.INFO)
nest_asyncio.apply()
load_dotenv()

stopwords = set([
    "the", "a", "an", "and", "or", "but", "about", "above", "after", "along", "amid", "among", "as", "at", "by", "for", "from", "in", "into", "like", "minus", "near", "of", "off", "on", "onto", "out", "over", "past", "per", "plus", "since", "till", "to", "under", "until", "up", "via", "vs", "with", "that", "can", "cannot", "could", "may", "might", "must", "need", "ought", "shall", "should", "will", "would", "have", "had", "has", "having", "be", "is", "am", "are", "was", "were", "being", "been", "get", "gets", "got", "gotten", "getting", "seem", "seeming", "seems", "seemed", "less", "least", "many", "more", "most", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
])

def extract_keywords(text: str, top_n: int = 5) -> List[str]:
    """
    Extract keywords from a given text.
    """
    words = re.findall(r'\w+', text.lower())
    filtered_words = [word for word in words if word not in stopwords and len(word) > 2]

    return [word for word, _ in Counter(filtered_words).most_common(top_n)]


def process_jsonl_data(datapath: str, 
                       **kwargs) -> List[Dict]:
    """
    Process jsonl data.
    """
    assert os.path.exists(datapath), "Data path does not exist"
    file_type = datapath.split(".")[-1]
    assert file_type == "jsonl", "Data file must be in jsonl format"
    splitter = SentenceSplitter(separator=".")
    split_fn = splitter._get_splits_by_fns
    
    data = []
    with open(datapath, 'r') as f:
        for line in f:
            line = json.loads(line)
            for doc in line['documents']:
                    data.append(Document(text=doc['text']))

    print(f"Loaded {len(data)} documents")

    chunks = splitter.get_nodes_from_documents(data, show_progress=True)

    print(f"Finish splitting {len(chunks)} chunks")

    processed_chunks = []
    for i, chunk in tqdm(enumerate(chunks), total=len(chunks), desc="Processing chunks"):
        processed_chunks.append({
            "text": chunk.text,
            "metadata": {
                "id": i,
                "keywords": extract_keywords(chunk.text),
                "length": len(chunk.text)
            }
        })

    print(f"Processed {len(processed_chunks)} chunks")
    return processed_chunks

if __name__ == "__main__":
    pass