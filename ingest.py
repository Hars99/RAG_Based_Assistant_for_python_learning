# ingest.py
import os
import uuid
from typing import List
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb

# Config
PDF_PATH = r"E:/projects/RAG_Assistant/data/Python_Tutorial_EDIT.pdf"  # update path if needed
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "python_tutorial"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def read_pdf_pages(path: str) -> List[str]:
    """Extract text from each page of the PDF"""
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text and text.strip():
            pages.append(text)
    return pages

def chunk_pages(pages: List[str], chunk_size: int = 700, chunk_overlap: int = 100):
    """Split pages into overlapping chunks"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = []
    for i, page_text in enumerate(pages):
        chunks = splitter.split_text(page_text)
        for j, chunk in enumerate(chunks):
            docs.append({
                "page_content": chunk,
                "metadata": {"page": i + 1, "chunk": j}
            })
    return docs

def embed_texts(texts: List[str], model_name: str = EMBED_MODEL):
    """Compute embeddings for text chunks"""
    embedder = SentenceTransformer(model_name)
    embeddings = embedder.encode(texts, show_progress_bar=True)
    embeddings = [list(e) if isinstance(e, (list, tuple)) else e for e in embeddings]
    return embeddings

def main():
    assert os.path.exists(PDF_PATH), f"PDF not found at {PDF_PATH}"

    # Step 1: Read PDF
    pages = read_pdf_pages(PDF_PATH)
    print(f"Read {len(pages)} pages from {PDF_PATH}")

    # Step 2: Chunk text
    docs = chunk_pages(pages, chunk_size=700, chunk_overlap=100)
    texts = [d["page_content"] for d in docs]
    metadatas = [d["metadata"] for d in docs]
    ids = [str(uuid.uuid4()) for _ in texts]
    print(f"Split into {len(texts)} chunks")

    # Step 3: Embed
    embeddings = embed_texts(texts)
    print("Embeddings computed.")

    # Step 4: Save to Chroma
    client = chromadb.PersistentClient(path=PERSIST_DIR)

    # Drop existing collection (if any)
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted old collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    # Create collection and add
    collection = client.create_collection(name=COLLECTION_NAME)

    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings
    )

    print(f"Ingested {len(texts)} chunks into collection '{COLLECTION_NAME}' at {PERSIST_DIR}")

if __name__ == "__main__":
    main()
