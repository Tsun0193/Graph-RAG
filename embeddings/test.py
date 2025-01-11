from LocalEmbedding import LocalEmbedding

embedder = LocalEmbedding()
sample = ["This is a sample text", "This is another sample text"]
embedding = embedder._get_text_embeddings(sample)
print(len(embedding), len(embedding[0]))