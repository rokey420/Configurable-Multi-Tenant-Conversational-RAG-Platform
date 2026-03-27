# backend/retrieval/pinecone_hybrid.py
import os
from pathlib import Path
from typing import List, Tuple

from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever


def _bm25_path_for_namespace(namespace: str) -> Path:
    """
    Store one BM25 per topic namespace so vocab stays correct per topic.
    """
    base = Path(os.getenv("BM25_DIR", "bm25"))
    base.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in namespace)
    return base / f"{safe}.json"


def load_or_fit_bm25(namespace: str, corpus_texts: List[str]) -> BM25Encoder:
    """
    Load persisted BM25 if available; otherwise fit on corpus_texts and persist.
    """
    path = _bm25_path_for_namespace(namespace)
    bm25 = BM25Encoder()

    if path.exists():
        bm25.load(str(path))
        return bm25

    if not corpus_texts:
        # If no corpus given, still return (but hybrid will be weak)
        return bm25

    bm25.fit(corpus_texts)
    bm25.dump(str(path))
    return bm25


def build_hybrid_retriever(index_name: str, embeddings, namespace: str, bm25: BM25Encoder) -> PineconeHybridSearchRetriever:
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(index_name)

    retriever = PineconeHybridSearchRetriever(
        embeddings=embeddings,
        sparse_encoder=bm25,
        index=index,
        namespace=namespace,   # ✅ topic isolation
    )
    return retriever