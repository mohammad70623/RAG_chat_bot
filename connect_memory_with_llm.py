import os 
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint

#step 1: setup LLM (Mistral with huggingface)
HF_TOKEN = os.environ.get("HF_TOKEN")
huggingface_repo_id = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id =huggingface_repo_id,
        temperature = 0.5 ,
        model_kwargs = {
            "token": HF_TOKEN,
            "max_length": 512}
    )
    return llm 


