import os 
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.pgvector import PGVector
from langchain.chains import retrieval_qa
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

file_paths = [
    os.path.join("data/",file) for file in os.listdir("data/") if file.endswith(".pdf") and os.path.isfile(os.path.join("data/"))
]

for file_path in file_paths:
    loader = PyPDFLoader(file_path)
    doc = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=100, add_start_index=True,separators=["\n", ".", "!", "?", ",", " "]
    )
    all_splits = text_splitter.split_documents(doc)
