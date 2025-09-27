from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 1. Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Sentences: positives & negatives
sentences = [
    # Refund / Return theme
    "I want a refund for my order",
    "What is the return policy for damaged items?",
    "How can I get my money back?",
    "The shop allows returns within 30 days",

    # Food theme
    "I love eating bananas",
    "This banana bread recipe is delicious",
    "How do you cook pasta?",
    "The chef prepared an amazing dish",

    # Travel theme
    "I booked a flight to New York",
    "The train departs tomorrow morning",
    "How can I cancel my hotel reservation?",
    "Travel insurance covers lost luggage"
]

# 3. Encode sentences
embeddings = model.encode(sentences)

# 4. Dimensionality reduction (t-SNE to 2D for visualization)
tsne = TSNE(n_components=2, random_state=42, perplexity=5, init="random")
embeddings_2d = tsne.fit_transform(embeddings)

# 5. Plot clusters
plt.figure(figsize=(12, 8))

# assign colors to 3 themes: refund, food, travel
colors = ["red"]*4 + ["green"]*4 + ["blue"]*4  

for i, (x, y) in enumerate(embeddings_2d):
    plt.scatter(x, y, c=colors[i], s=100)
    plt.text(x+0.05, y+0.05, sentences[i], fontsize=9)

plt.title("t-SNE Visualization of Sentence Embeddings (Semantic Clusters)")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()
