from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# STEP 1: Load PDF
loader = PyPDFLoader("RAG/data/saurabh-bharti-resume.pdf")
docs = loader.load()
print(f"\n=== STEP 1: Loaded {len(docs)} pages from PDF")

# STEP 2: Chunking
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(f"=== STEP 2: Split into {len(chunks)} chunks")

# STEP 3: Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# STEP 4: Store in Vector DB
vectorDb = Chroma.from_documents(chunks, embedding=embeddings, persist_directory="./chroma_db")
retriever = vectorDb.as_retriever(search_kwargs={"k": 3})

# STEP 5: LLM Setup
llm = OllamaLLM(model="gemma3:4b")

# STEP 6: Custom Prompt Template
custom_prompt = PromptTemplate.from_template("""
You are an assistant. Use ONLY the provided context to answer the question.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}
Answer:
""")

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": custom_prompt}
)

# STEP 7: Query Loop with Debugging
print("\n=== RAG System Ready ::")
while True:
    query = input("\nAsk a question (or type 'exit'): ")
    if query.lower() == "exit":
        print("Exiting.")
        break

    # Debug: show retrieved docs
    retrieved_docs = retriever.invoke(query)
    
    # Debug: build prompt manually
    context_text = "\n".join([d.page_content for d in retrieved_docs])
    final_prompt = custom_prompt.format(context=context_text, question=query)

    # Run QA chain
    result = qa.invoke(query)
    print("\n--- LLM Answer ---")
    print(result["result"])


'''
Results on Resume Reader

Ask a question (or type 'exit'): what is his experience?

--- LLM Answer ---
He worked on PCF-hosted web applications, designed and developed product features, and built Microservices for a chatbot facilitating password resets and account management. He has experience with Full Stack Java, MERN stack, Spring Boot, RESTful web services, Microservices, and Event-Driven Solutions.

Ask a question (or type 'exit'): what was done in boeing training library?

--- LLM Answer ---
Implemented and optimized core features for a secure, cloud-native content distribution system with seamless upload/download workflows and a proof-of-concept video streaming module.

Ask a question (or type 'exit'): 
'''