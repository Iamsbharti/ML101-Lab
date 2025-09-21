from openai import OpenAI
import numpy as np

'''

Embedding is a way to turn text (word, sentencesa ,documents) into a long list of numbers (a vector)
that captures the meaning of the text.

An embedding model like `text-embedding-3-small` from openai would turn sentences like
input :  i want a refund for my order.
output:  [-0.002491552382707596, -0.004974357318133116, -0.08066273480653763, -0.029647869989275932,...]

If we do the same for the query `how do i get my money back?`, we get another vector , then we
compute cosine similarity (angler between vectors) , if the angle is small (close to 1.0), its semantically close.
'''
client = OpenAI()

def cosine_similarity(vector1,vector2):
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

sentences = [ 
    "i want a refund for my order.",
    "how can i get my money back.",
    "who are you??",
    "bananas are yellow and sweet",
    "i love you",
    "you mean a lot to me.",
    "do i love you?"
]

# user openai embedding model
embeddings = [client.embeddings.create(input=s, model="text-embedding-3-small").data[0].embedding for s in sentences]

simliarity_1 = cosine_similarity(embeddings[0], embeddings[1])
simliarity_2 = cosine_similarity(embeddings[0], embeddings[2])
simliarity_3 = cosine_similarity(embeddings[4], embeddings[5])
simliarity_4 = cosine_similarity(embeddings[4], embeddings[6])


print(len(embeddings[0])) #1536 dimensions

print("refund vs money back:",simliarity_1)
print("refund vs who are you:",simliarity_2)

print("love vs meaning:",simliarity_3)
print("love vs love?:",simliarity_4)

'''
Results:

1536
refund vs money back: 0.5826366951753005
refund vs who are you: 0.1047923857243988
love vs meaning: 0.47592194708822444
love vs love?: 0.6668276089984836

'''