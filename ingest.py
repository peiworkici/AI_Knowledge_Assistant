
import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

DATA_PATH = "data/handbook.pdf"
DB_PATH = "chroma_db"

def main():
    # 1. Load PDF
    loader = PyPDFLoader(DATA_PATH)
    documents = loader.load()

    print(f"Loaded {len(documents)} pages")

    # 2. Split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks")

    # 3. Create embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    # 4. Store chunks in Chroma
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )

    print("Documents indexed successfully!")

if __name__ == "__main__":
    main()
