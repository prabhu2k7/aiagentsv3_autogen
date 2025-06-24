# rag_app/ingest.py

import os
from dotenv import load_dotenv

# 1) Load .env so OPENAI_API_KEY is available
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in your .env")

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Compute paths relative to this file’s parent directory (rag_app/)
PROJECT_ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONCOLOGY_PDF        = os.path.join(PROJECT_ROOT, "oncology.pdf")
NEUROLOGY_PDF       = os.path.join(PROJECT_ROOT, "neurology.pdf")
ONCOLOGY_INDEX_DIR  = os.path.join(PROJECT_ROOT, "oncology_index")
NEUROLOGY_INDEX_DIR = os.path.join(PROJECT_ROOT, "neurology_index")

def ingest_pdf_to_faiss(pdf_path: str, index_dir: str):
    """
    Load a PDF, split into chunks, embed with OpenAI, build & save FAISS index.
    """
    print(f"Ingesting {os.path.basename(pdf_path)} → {index_dir}")
    loader    = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"  Loaded {len(documents)} pages")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks   = splitter.split_documents(documents)
    print(f"  Split into {len(chunks)} chunks")

    emb   = OpenAIEmbeddings(openai_api_key=API_KEY)
    index = FAISS.from_documents(chunks, emb)

    os.makedirs(index_dir, exist_ok=True)
    index.save_local(index_dir)
    print(f"  Saved FAISS index to {index_dir}\n")

def ingest_all():
    """
    Ensure both oncology_index/ and neurology_index/ exist,
    ingesting from the PDFs if they do not.
    """
    for pdf, idx_dir in [
        (ONCOLOGY_PDF,  ONCOLOGY_INDEX_DIR),
        (NEUROLOGY_PDF, NEUROLOGY_INDEX_DIR),
    ]:
        idx_file = os.path.join(idx_dir, "index.faiss")
        if not os.path.exists(idx_file):
            ingest_pdf_to_faiss(pdf, idx_dir)
        else:
            print(f"Index already exists at {idx_file}, skipping ingest")

if __name__ == "__main__":
    ingest_all()
    print("✅ Ingestion complete!")
