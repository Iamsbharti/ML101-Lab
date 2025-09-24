from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# =====================================================
# STEP 1: Load PDF
# =====================================================
loader = PyPDFLoader("RAG/data/The elements of data analytics.pdf")
docs = loader.load()
print(f"\n=== STEP 1: Loaded {len(docs)} pages from PDF ===")

# =====================================================
# STEP 2: Chunking
# =====================================================
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(f"=== STEP 2: Split into {len(chunks)} chunks ===")

# print("\n--- Sample Chunks ---")
# for i, c in enumerate(chunks[:90]):
#     print(f"\nChunk {i+1}:\n{c.page_content[:300]}...\n")

# =====================================================
# STEP 3: Embeddings
# =====================================================
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# =====================================================
# STEP 4: Store in Vector DB
# =====================================================
vectorDb = Chroma.from_documents(chunks, embedding=embeddings, persist_directory="./chroma_db")
retriever = vectorDb.as_retriever(search_kwargs={"k": 3})

# =====================================================
# STEP 5: LLM Setup
# =====================================================
llm = OllamaLLM(model="gemma3:4b")

# =====================================================
# STEP 6: Custom Prompt Template
# =====================================================
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

# =====================================================
# STEP 7: Query Loop with Debugging
# =====================================================
print("\n=== RAG System Ready ===")
while True:
    query = input("\nAsk a question (or type 'exit'): ")
    if query.lower() == "exit":
        print("Exiting.")
        break

    # Debug: show retrieved docs
    retrieved_docs = retriever.get_relevant_documents(query)
    # print("\n--- Retrieved Documents ---")
    # for i, d in enumerate(retrieved_docs, 1):
    #     print(f"\nDoc {i}:\n{d.page_content[:500]}...\n")

    # Debug: build prompt manually
    context_text = "\n".join([d.page_content for d in retrieved_docs])
    final_prompt = custom_prompt.format(context=context_text, question=query)
    print("\n--- Final Prompt Sent to LLM ---")
    print(final_prompt)

    # Run QA chain
    result = qa.invoke(query)
    print("\n--- LLM Answer ---")
    print(result["result"])
