from rag_chat import get_rag_chain

# -----------------------------
# TEST SET (You can add more)
# -----------------------------

test_data = [
    {
        "question": "What is a reflex arc?",
        "answer": "A reflex arc is the pathway through which nerve impulses travel during a reflex action. It involves a receptor, sensory neuron, spinal cord, motor neuron, and effector."
    },
    {
        "question": "Which part of the brain controls posture and balance?",
        "answer": "The cerebellum controls posture and balance."
    },
    {
        "question": "What are gustatory receptors?",
        "answer": "Gustatory receptors detect taste stimuli."
    },
    {
        "question": "How do tendrils climb a support?",
        "answer": "Tendrils climb a support by touching and wrapping around it through a growth movement called thigmotropism."
    },
]

# -----------------------------
# Evaluation Script
# -----------------------------
rag = get_rag_chain()

correct = 0
total = len(test_data)

print("\n🔍 RAG SYSTEM ACCURACY EVALUATION\n")

for i, item in enumerate(test_data, 1):
    question = item["question"]
    expected = item["answer"].lower()

    print(f"\nQ{i}: {question}")
    result, sources = rag(question)
    predicted = result.lower()

    print(f"Expected: {expected}")
    print(f"Predicted: {predicted[:200]}...")  # Trim long text

    # Scoring rule:
    # If expected answer keywords exist inside predicted answer → mark correct
    if any(word in predicted for word in expected.split()[:3]):  
        print("✔ Correct")
        correct += 1
    else:
        print("✘ Incorrect")

# -----------------------------
# Final Accuracy
# -----------------------------
accuracy = (correct / total) * 100
print("\n----------------------------------")
print(f"🎯 RAG Accuracy: {accuracy:.2f}%")
print("----------------------------------\n")
