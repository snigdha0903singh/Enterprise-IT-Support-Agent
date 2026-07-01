import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from ingestion.build_vectorstore import COLLECTION_NAME, build_vectorstore, get_qdrant_client
from ingestion.chunk_documents import chunk_documents
from ingestion.embed_documents import DEFAULT_EMBEDDING_MODEL, embed_documents
from ingestion.load_documents import load_all_documents
from ingestion.metadata_builder import add_metadata_to_documents


DEFAULT_QDRANT_PATH = REPO_ROOT / "vectorstore" / "qdrant"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full TechCorp ingestion pipeline.")
    parser.add_argument(
        "--collection",
        default=COLLECTION_NAME,
        help="Qdrant collection name.",
    )
    parser.add_argument(
        "--qdrant-url",
        default=None,
        help="Qdrant server URL. If omitted, local Qdrant storage is used.",
    )
    parser.add_argument(
        "--qdrant-path",
        default=str(DEFAULT_QDRANT_PATH),
        help="Local Qdrant storage path used when --qdrant-url is omitted.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="SentenceTransformer model name.",
    )
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--upsert-batch-size", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("1. Loading markdown documents...")
    documents = load_all_documents()
    print(f"   Loaded {len(documents)} documents")

    print("2. Building metadata...")
    documents = add_metadata_to_documents(documents)
    print("   Metadata added")

    print("3. Chunking documents...")
    chunks = chunk_documents(
        documents,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(f"   Created {len(chunks)} chunks")

    print("4. Embedding chunks...")
    embedded_documents = embed_documents(
        chunks,
        model_name=args.embedding_model,
        batch_size=args.embedding_batch_size,
    )
    print(f"   Embedded {len(embedded_documents)} chunks")

    print("5. Storing vectors in Qdrant...")
    
    client = get_qdrant_client(url=args.qdrant_url, path=None if args.qdrant_url else args.qdrant_path)
    print(f"   Using Qdrant client {client}")
    build_vectorstore(
        embedded_documents,
        collection_name=args.collection,
        client=client,
        batch_size=args.upsert_batch_size,
    )

    location = args.qdrant_url or args.qdrant_path
    print(f"Done. Collection '{args.collection}' is stored in Qdrant at {location}")


if __name__ == "__main__":
    main()
