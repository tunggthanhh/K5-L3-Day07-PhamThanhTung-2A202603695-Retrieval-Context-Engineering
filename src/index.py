"""Create the persistent Day 7 dense index from exported chunks."""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
from dotenv import load_dotenv
from .day6.embeddings import MyEmbedder
from .day6.models import Chunk
from .day6.store import QdrantLocalStore

ROOT = Path(__file__).resolve().parents[1]

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--chunks", type=Path, default=ROOT / "artifacts/data/chunks.json")
    parser.add_argument("--store", type=Path, default=ROOT / "artifacts/qdrant"); args = parser.parse_args()
    load_dotenv(ROOT / ".env"); model = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
    rows = json.loads(args.chunks.read_text(encoding="utf-8")); embedder = MyEmbedder(model)
    chunks = [Chunk(chunk_id=x["id"], document_id=x["source_id"], source=x["source"], page=x["page"], text=x["text"], metadata=x["metadata"]) for x in rows]
    store = QdrantLocalStore(args.store, embedder.dimensions)
    try: store.upsert("paragraph", chunks, embedder.embed_documents([x.text for x in chunks]))
    finally: store.close()
    print(f"Indexed {len(chunks)} chunks with {model} into {args.store}")

if __name__ == "__main__": main()
