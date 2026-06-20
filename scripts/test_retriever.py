import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from retrieval.retriever import DEFAULT_QDRANT_PATH, get_retriever


QUERIES = [
    "Who owns Polaris Reporting?",
    "Who is Priya Nayar?",
    "What approvals are required for Remember Tracker?",
    "Which systems does Alexander Collins use?",
    "Who is the backup owner of Polaris Reporting?",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
            print(doc.page_content[:300])


if __name__ == "__main__":
    main()
