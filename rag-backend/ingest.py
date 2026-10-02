"""Build FAISS index from ../data. Run: python ingest.py"""
from rag import build_index

if __name__ == "__main__":
    info = build_index()
    print(f"Indexed {info['docs']} chunks, dim={info['dim']}, backend={info['backend']}")
