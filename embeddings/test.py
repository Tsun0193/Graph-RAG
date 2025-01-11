from LocalEmbedding import LocalEmbedding

embedder = LocalEmbedding()
sample = "This is a sample text"
embedding = embedder.get_text_embedding(sample)
print(embedding)