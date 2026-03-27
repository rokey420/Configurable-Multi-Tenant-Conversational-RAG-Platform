import os
import uuid
import argparse
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone


def load_env():
    project_root = Path(__file__).resolve().parent.parent
    env_path = project_root / ".env"
    load_dotenv(dotenv_path=env_path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--namespace", required=True, help="Pinecone namespace (e.g., topic-hr-... or general)")
    p.add_argument("--data", required=True, help="Folder path containing .pdf/.txt files to ingest")
    return p.parse_args()


def main():
    load_env()
    args = parse_args()

    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ragbot-index")

    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY missing in .env")

    namespace = args.namespace.strip()
    if not namespace:
        raise RuntimeError("namespace cannot be empty")

    data_path = Path(args.data).expanduser().resolve()
    if not data_path.exists():
        raise RuntimeError(f"DATA path not found: {data_path}")

    print("Loading docs from:", str(data_path))
    txt_loader = DirectoryLoader(
        str(data_path),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    pdf_loader = DirectoryLoader(
        str(data_path),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
    )

    docs = txt_loader.load() + pdf_loader.load()
    print(f"Loaded {len(docs)} raw documents/pages.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    texts, metadatas = [], []
    for i, d in enumerate(chunks):
        text = d.page_content or ""
        md = d.metadata or {}

        raw_source = md.get("source") or md.get("file_path") or md.get("filename") or "unknown"
        source = Path(str(raw_source)).name
        page = md.get("page", None)

        meta = {"text": text, "source": source, "chunk_id": i}
        if page is not None:
            try:
                meta["page"] = int(page)
            except Exception:
                pass

        texts.append(text)
        metadatas.append({k: v for k, v in meta.items() if v is not None})

    print("Loading embedding model...")
    emb_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Embedding all chunks...")
    dense_vectors = emb_model.embed_documents(texts)

    print("Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    BATCH_SIZE = 100
    to_upsert = []

    print(f"Upserting to Pinecone namespace: {namespace}")
    for i in range(len(texts)):
        item = {
            "id": str(uuid.uuid4()),
            "values": dense_vectors[i],
            "metadata": metadatas[i],
        }
        to_upsert.append(item)

        if len(to_upsert) >= BATCH_SIZE:
            index.upsert(vectors=to_upsert, namespace=namespace)
            to_upsert = []

    if to_upsert:
        index.upsert(vectors=to_upsert, namespace=namespace)

    print("✅ Ingestion complete!")
    print(f"✅ Index: {PINECONE_INDEX_NAME}")
    print(f"✅ Namespace: {namespace}")


if __name__ == "__main__":
    main()