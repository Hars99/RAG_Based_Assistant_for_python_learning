
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np

class QueryProcessor:
    def __init__(self, db_path="./chroma_db", model_name="all-MiniLM-L6-v2", top_k=3):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection("python_tutorial")
        self.model = SentenceTransformer(model_name)
        self.top_k = top_k

    def embed_query(self, query: str):
        return self.model.encode([query])[0]

    def retrieve_context(self, query: str):
        query_vector = self.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=self.top_k
        )
        return results["documents"][0], results["metadatas"][0]

if __name__ == "__main__":
    qp = QueryProcessor()
    docs, meta = qp.retrieve_context("What is a decorator in Python?")
    for d, m in zip(docs, meta):
        print(f"[Page {m['page']}] {d[:150]}...")
