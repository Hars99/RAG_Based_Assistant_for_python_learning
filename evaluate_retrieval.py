
from query_processing import QueryProcessor
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Example test set (you can expand this)
test_queries = [
    {"query": "Explain Python decorators", "expected_answer": "function wrappers"},
    {"query": "What is a list comprehension?", "expected_answer": "compact way to create lists"},
]

def evaluate_retrieval():
    qp = QueryProcessor()
    model = SentenceTransformer("all-MiniLM-L6-v2")

    similarities = []
    for test in test_queries:
        docs, _ = qp.retrieve_context(test["query"])
        retrieved_text = " ".join(docs)
        sim = util.cos_sim(
            model.encode(test["expected_answer"]),
            model.encode(retrieved_text)
        ).item()
        similarities.append(sim)
        print(f"Query: {test['query']}")
        print(f"→ Similarity: {sim:.3f}\n")

    avg_score = np.mean(similarities)
    print(f"Average Retrieval Similarity: {avg_score:.3f}")

if __name__ == "__main__":
    evaluate_retrieval()
