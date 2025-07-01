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

document_list = [
    file for file in os.listdir("data/") if file.endswith(".pdf") and os.path.isfile(os.path.join("data/"))
]

for document in document_list:


