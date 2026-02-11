import torch
from sentence_transformers import SentenceTransformer, util
from rag_chat import get_rag_chain

# Load RAG pipeline
rag = get_rag_chain()

# Semantic embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# ---------------------------
# Test set (You can expand)
# ---------------------------
test_data = [
    {
        "question": "What is a reflex arc?",
        "answer": "A reflex arc is the pathway through which nerve impulses travel during a reflex action.",
    },
    {
        "question": "Which part of the brain controls posture and balance?",
        "answer": "The cerebellum controls posture and balance.",
    },
    {
        "question": "What are gustatory receptors?",
        "answer": "Gustatory receptors detect taste stimuli.",
    },
    {
        "question": "How do tendrils climb a support?",
        "answer": "Tendrils climb a support by wrapping around an object through thigmotropism.",
    },
]


# ---------------------------
# Semantic Evaluation Logic
# ---------------------------
def semantic_similarity(expected, predicted):
    emb1 = model.encode(expected, convert_to_tensor=True)
    emb2 = model.encode(predicted, convert_to_tensor=True)
    return float(util.cos_sim(emb1, emb2))


def evaluate():
    print("\n🔍 SEMANTIC RAG EVALUATION\n")

    total = len(test_data)
    scores = []

    for i, item in enumerate(test_data, 1):
        q = item["question"]
        expected = item["answer"]

        # Run RAG
        predicted, sources = rag(q)

        # Compute semantic similarity
        sim = semantic_similarity(expected, predicted)

        scores.append(sim)

        print(f"\nQ{i}: {q}")
        print(f"Expected: {expected}")
        print(f"Predicted: {predicted[:200]}...\n")
        print(f"⭐ Semantic Similarity: {sim:.4f}")

    # Final Score
    avg = sum(scores) / total
    print("\n-------------------------------------------")
    print(f"🎯 FINAL SEMANTIC ACCURACY: {avg*100:.2f}%")
    print("-------------------------------------------\n")


if __name__ == "__main__":
    evaluate()
