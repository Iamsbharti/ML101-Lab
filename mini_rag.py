from openai import OpenAI
import numpy as np

client = OpenAI()

DOCS = [
    "This is a refund policy document",
    "Company name is neo shopping",
    "Our refund policy allows returns within 30 days of purchase provided the item is in original condition.",
    "Customers can request a refund by contacting support and providing the order number.",
    "We offer free shipping for orders above $50 and standard rates otherwise.",
    "Warranty claims must be submitted within one year and include proof of purchase.",
    "To exchange an item, return it in unused condition and place a new order or contact our support."
]

# create embeddings
def get_embeddings(text:str) -> list[float]:
    response = client.embeddings.create(input = text,model="text-embedding-3-small")
    return response.data[0].embedding

# compute cosine similarity (angle between vectors)
def cosine_similarity(a, b)-> float:
    return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))


# precompute embeddings for all the documents
doc_embeddings = [get_embeddings(d) for d in DOCS]

# query loop
while True:
    query = input("\nAsk me something (or type 'exit'): ")
    if(query.lower() == 'exit'):
        break

    # embed query
    q_emb = get_embeddings(query)

    # compute similarity
    similarity = [cosine_similarity(q_emb,d_emb) for d_emb in doc_embeddings]

    # rank top-k
    k = 3
    ranked = sorted(zip(DOCS, similarity), key = lambda x:x[1], reverse=True)[:k]

    # print results
    print("\n Top Matches:")
    for i, (doc,score) in enumerate(ranked,1):
        print(f"{i}. (score={score:.3f}) {doc}")
