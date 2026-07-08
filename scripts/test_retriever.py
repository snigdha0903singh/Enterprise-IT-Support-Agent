import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from retrieval.retriever import DEFAULT_QDRANT_PATH, get_retriever

QUERIES = [
    "Aimee Montoya",
    "What should I do for application slow performance?",
    "How do I handle a credential stuffing attack?",
    "What is the escalation path for Despite Manager?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test retrieval against the TechCorp vector store.")
    parser.add_argument("--k", type=int, default=5, help="Number of documents to retrieve.")
    parser.add_argument("--qdrant-url", default=None, help="Qdrant server URL.")
    parser.add_argument(
        "--qdrant-path",
        default=str(DEFAULT_QDRANT_PATH),
        help="Local Qdrant storage path used when --qdrant-url is omitted.",
    )

    parser.add_argument(
        "--reingest",
        action="store_true",
        help="Run the full ingestion pipeline before testing retrieval.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=int(os.getenv("CHUNK_SIZE", 800)),
        help="Chunk size in characters used with --reingest.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=int(os.getenv("CHUNK_OVERLAP", 200)),
        help="Chunk overlap in characters used with --reingest.",
    )

    return parser.parse_args()


def run_command(cmd: list[str]) -> None:
    print(f"Running: {' '.join(cmd)}")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"Command failed: {' '.join(cmd)} (exit {res.returncode})")
        sys.exit(res.returncode)


def main() -> None:
    args = parse_args()

    if args.reingest:
        ingest_cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "ingest.py"),
            "--chunk-size",
            str(args.chunk_size),
            "--chunk-overlap",
            str(args.chunk_overlap),
        ]
        if args.qdrant_url:
            ingest_cmd += ["--qdrant-url", args.qdrant_url]
        else:
            ingest_cmd += ["--qdrant-path", args.qdrant_path]
        run_command(ingest_cmd)

    retriever = get_retriever(
        k=args.k,
        url=args.qdrant_url,
        path=None if args.qdrant_url else args.qdrant_path,
    )

    for query in QUERIES:
        print(f"\nQUERY: {query}")

        docs = retriever.invoke(query)

        for index, doc in enumerate(docs):
            print("=" * 50)
            print(f"Rank {index + 1}")
            print(doc.metadata)
            print(doc.page_content[:700])

if __name__ == "__main__":
    main()
