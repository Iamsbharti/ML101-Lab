import re
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI


# STEP 1: load Documents
def load_documents():
    docs = {
        "policy.txt": """
        Our refund policy allows returns within 30 days of purchase.
        Items must be in original condition with proof of purchase.
        Warranty claims must be submitted within one year.
        """,
        "shipping.txt": """
        We offer free shipping for orders above $50.
        Standard shipping rates apply otherwise.
        Orders are processed within 2 business days.
        """
    }
    return docs

print("\n=== STEP 1 & 2: Loading documents and chunked into passages===")
docs = load_documents()
chunks = []


# STEP 2: chunking
def chunk_texts(text, chunk_size=50):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

for fname, content in docs.items():
    for chunk in chunk_texts(content):
        chunks.append({"doc": fname, "text": chunk})

for c in chunks:
    print(f" - From {c['doc']}: {c['text']}")


# STEP 3: embeddings
print("\n=== STEP 3: Creating embeddings ===")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

for c in chunks:
    c["embedding"] = embedder.encode(c["text"], convert_to_numpy=True)

print(f"Created embeddings. Example vector length: {len(chunks[0]['embedding'])}")

# STEP 4: retrieval (k-NN with cosine similarity)
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve(query, chunks, k=3):
    # encode the query
    q_embed = embedder.encode(query, convert_to_numpy=True)
    scored = []
    # compute the cosine similarity by itrating over chunks and w.r.t. computed embeding value
    for c in chunks:
        score = float(cosine_similarity(q_embed, c["embedding"]))
        scored.append((score, c))
    # sort the score in reverse
    ranked = sorted(scored, key=lambda x: x[0], reverse=True)
    # return top 3 ranked passages
    return ranked[:k]

# STEP 5: prompt assembly
def build_prompt(query, retrieved_chunks):
    # create context bsaed on the passage under 'text'
    context = "\n".join([f"- {c['text']}" for _, c in retrieved_chunks])
    prompt = f"""
You are a helpful assistant. Use the context below to answer the question.
If the answer is not in the context, say 'I don't know'.

Context:
{context}

Question: {query}
Answer:"""
    return prompt


# STEP 6: query Loop with LLM Generation
print("\n=== RAG System Ready ===")
print("Type your questions below (type 'exit' to quit).")

client = OpenAI()

while True:
    query = input("\nAsk me something: ")
    if query.lower() == "exit":
        print("Exiting SCRATCH RAG system. Goodbye!")
        break

    print("\n--- STEP 4: Retrieval ---")
    top_k = retrieve(query, chunks)
    for score, c in top_k:
        print(f"[{c['doc']}] (score={score:.3f}) {c['text']}")

    print("\n--- STEP 5: Prompt Assembly ---")
    prompt = build_prompt(query, top_k)
    print(prompt)

    print("\n--- STEP 6: LLM Generation ---")
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = resp.choices[0].message.content
    print("LLM Answer:", answer)


'''
results
=== STEP 1 & 2: Loading documents and chunked into passages===
 - From policy.txt: Our refund policy allows returns within 30 days of purchase. Items must be in original condition with proof of purchase. Warranty claims must be submitted within one year.
 - From shipping.txt: We offer free shipping for orders above $50. Standard shipping rates apply otherwise. Orders are processed within 2 business days.

=== STEP 3: Creating embeddings ===
Created embeddings. Example vector length: 384

=== RAG System Ready ===
Type your questions below (type 'exit' to quit).

Ask me something: what are standard rates?

--- STEP 4: Retrieval ---
[shipping.txt] (score=0.296) We offer free shipping for orders above $50. Standard shipping rates apply otherwise. Orders are processed within 2 business days.
[policy.txt] (score=0.082) Our refund policy allows returns within 30 days of purchase. Items must be in original condition with proof of purchase. Warranty claims must be submitted within one year.

--- STEP 5: Prompt Assembly ---

You are a helpful assistant. Use the context below to answer the question.
If the answer is not in the context, say 'I don't know'.

Context:
- We offer free shipping for orders above $50. Standard shipping rates apply otherwise. Orders are processed within 2 business days.
- Our refund policy allows returns within 30 days of purchase. Items must be in original condition with proof of purchase. Warranty claims must be submitted within one year.

Question: what are standard rates?
Answer:

--- STEP 6: LLM Generation ---
LLM Answer: I don't know.

Ask me something: what is proff of purchase?

--- STEP 4: Retrieval ---
[policy.txt] (score=0.194) Our refund policy allows returns within 30 days of purchase. Items must be in original condition with proof of purchase. Warranty claims must be submitted within one year.
[shipping.txt] (score=0.171) We offer free shipping for orders above $50. Standard shipping rates apply otherwise. Orders are processed within 2 business days.

--- STEP 5: Prompt Assembly ---

You are a helpful assistant. Use the context below to answer the question.
If the answer is not in the context, say 'I don't know'.

Context:
- Our refund policy allows returns within 30 days of purchase. Items must be in original condition with proof of purchase. Warranty claims must be submitted within one year.
- We offer free shipping for orders above $50. Standard shipping rates apply otherwise. Orders are processed within 2 business days.

Question: what is proff of purchase?
Answer:

--- STEP 6: LLM Generation ---
LLM Answer: I don't know.

Ask me something: what are shipping charges?

--- STEP 4: Retrieval ---
[shipping.txt] (score=0.645) We offer free shipping for orders above $50. Standard shipping rates apply otherwise. Orders are processed within 2 business days.
[policy.txt] (score=0.101) Our refund policy allows returns within 30 days of purchase. Items must be in original condition with proof of purchase. Warranty claims must be submitted within one year.

--- STEP 5: Prompt Assembly ---

You are a helpful assistant. Use the context below to answer the question.
If the answer is not in the context, say 'I don't know'.

Context:
- We offer free shipping for orders above $50. Standard shipping rates apply otherwise. Orders are processed within 2 business days.
- Our refund policy allows returns within 30 days of purchase. Items must be in original condition with proof of purchase. Warranty claims must be submitted within one year.

Question: what are shipping charges?
Answer:

--- STEP 6: LLM Generation ---
LLM Answer: Standard shipping rates apply for orders below $50. For orders above $50, we offer free shipping.

Ask me something: exit 
Exiting SCRATCH RAG system. Goodbye!
'''