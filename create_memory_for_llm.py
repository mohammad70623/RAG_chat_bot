from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

#Step 1: Load the raw pdf
DATA_PATH = "data/"
def load_pdf_files(data): 
    loader = DirectoryLoader(data, 
                            glob = '*.pdf', 
                            loader_cls = PyPDFLoader)

    documents = loader.load() 
    return documents 
    
documents = load_pdf_files(data=DATA_PATH)
print("Length of pdf", len(documents))