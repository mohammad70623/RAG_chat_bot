import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ─── Step 1: Load PDF Files ────────────────────────────────────────────────────
DATA_PATH = "data/"

def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

documents = load_pdf_files(data=DATA_PATH)
print(f"✅ Loaded {len(documents)} document pages.")

# ─── Step 2: Create Chunks ─────────────────────────────────────────────────────
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks = create_chunks(extracted_data=documents)
print(f"✅ Created {len(text_chunks)} text chunks.")

# ─── Step 3: Embedding Model ───────────────────────────────────────────────────
def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embedding_model

embedding_model = get_embedding_model()
print("✅ Embedding model loaded.")

# ─── Step 4: Store in ChromaDB ────────────────────────────────────────────────
DB_CHROMA_PATH = "vectorstore/db_chroma"
os.makedirs(DB_CHROMA_PATH, exist_ok=True)

db = Chroma.from_documents(
    documents=text_chunks,
    embedding=embedding_model,
    persist_directory=DB_CHROMA_PATH,
    collection_name="medbot_collection"
)

print(f"✅ ChromaDB vector store saved at: {DB_CHROMA_PATH}")
print(f"✅ Total vectors stored: {db._collection.count()}")
