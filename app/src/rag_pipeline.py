"""Load -> Embed -> Search -> Generate."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import chromadb
import torch
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings

from src.load_esci import load_corpus

CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "constraintrag_v0"
BATCH_SIZE = 64


def get_embeddings() -> HuggingFaceEmbeddings:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 16},
    )


def build_index(embeddings, extra_products: int = 5000):
    """Reuse the persisted index if it exists; otherwise embed the corpus."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    if collection.count() > 0:
        return collection
    corpus = load_corpus(extra_products)
    for start in range(0, len(corpus), BATCH_SIZE):
        batch = corpus.iloc[start : start + BATCH_SIZE]
        documents = batch["document"].tolist()
        collection.add(
            ids=batch["product_id"].tolist(),
            documents=documents,
            embeddings=embeddings.embed_documents(documents),
            metadatas=[{"product_id": p, "title": t} for p, t in zip(batch["product_id"], batch["title"])],
        )
        print(f"indexed {min(start + BATCH_SIZE, len(corpus))}/{len(corpus)}", flush=True)
    return collection


def search(query: str, collection, embeddings, top_k: int = 10) -> list[tuple[float, Document]]:
    results = collection.query(query_embeddings=[embeddings.embed_query(query)], n_results=top_k)
    return [
        (1 - distance, Document(page_content=text, metadata=metadata))
        for text, metadata, distance in zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
    ]


def generate_answer(query: str, retrieved: list[tuple[float, Document]], llm) -> str:
    context = "\n\n".join(f"[{doc.metadata['product_id']}] {doc.page_content}" for _, doc in retrieved)
    response = llm.invoke([
        SystemMessage(content="Recommend products using only the context. Answer briefly. If nothing fits, say so."),
        HumanMessage(content=f"Context:\n{context}\n\nRequest: {query}"),
    ])
    return response.text.strip()
