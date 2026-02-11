import os
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


load_dotenv()

PDF_FOLDER = "data/pdfs/"
VECTOR_DB_DIR = "vectorstore/"

def load_pdfs():
    documents = []
    for file in os.listdir(PDF_FOLDER):
        if file.endswith(".pdf"):
            path = os.path.join(PDF_FOLDER, file)
            loader = PyPDFLoader(path)
            docs = loader.load()
            documents.extend(docs)
            print(f"[✓] Loaded: {file}")
    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)
    print(f"[✓] Created {len(chunks)} chunks.")
    return chunks

def create_vector_db(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(VECTOR_DB_DIR)
    print("[✓] Vector DB saved")

def main():
    print("🚀 Starting ingestion...")
    docs = load_pdfs()
    chunks = split_documents(docs)
    create_vector_db(chunks)
    print("🎉 Ingestion complete!")

if __name__ == "__main__":
    main()
