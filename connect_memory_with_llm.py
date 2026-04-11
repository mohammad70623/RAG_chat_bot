import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

# ─── Step 1: Load Environment & Groq LLM ──────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME = "llama-3.1-8b-instant"

def load_llm():
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name=GROQ_MODEL_NAME,
        temperature=0.5,
        max_tokens=512
    )
    return llm

# ─── Step 2: Custom Prompt Template ───────────────────────────────────────────
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know.
Don't try to make up an answer.
Don't provide anything outside the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )
    return prompt

# ─── Step 3: Load ChromaDB Vector Store ───────────────────────────────────────
DB_CHROMA_PATH = "vectorstore/db_chroma"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory=DB_CHROMA_PATH,
    embedding_function=embedding_model,
    collection_name="medbot_collection"
)
print(f"✅ ChromaDB loaded. Total vectors: {db._collection.count()}")

# ─── Step 4: Build Retrieval QA Chain ─────────────────────────────────────────
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={
        "prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)
    }
)

# ─── Step 5: Query ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    user_query = input("Write your query here: ")
    response   = qa_chain.invoke({"query": user_query})

    print("\n" + "─" * 60)
    print("RESULT:\n", response["result"])
    print("\nSOURCE DOCUMENTS:")
    for i, doc in enumerate(response["source_documents"], 1):
        print(f"\n[{i}] {doc.metadata.get('source', 'Unknown')} — Page {doc.metadata.get('page', 'N/A')}")
        print(doc.page_content[:300])
    print("─" * 60)
