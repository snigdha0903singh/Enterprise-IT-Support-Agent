import os
from collections.abc import Sequence
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


COLLECTION_NAME = "techcorp"
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_QDRANT_PATH = "vectorstore/qdrant"


def get_qdrant_client(
    url: str | None = None,
    path: str | None = None,
    api_key: str | None = None,
) -> QdrantClient:
    qdrant_path = path or os.getenv("QDRANT_PATH")

    if qdrant_path:
        return QdrantClient(path=qdrant_path)

    return QdrantClient(
        url=url or os.getenv("QDRANT_URL", DEFAULT_QDRANT_URL),
        api_key=api_key or os.getenv("QDRANT_API_KEY"),
    )


def _vector_size(embedded_documents: Sequence[dict[str, Any]]) -> int:
    if not embedded_documents:
        raise ValueError("No embedded documents provided.")

    vector = embedded_documents[0].get("vector")
    if not vector:
        raise ValueError("Embedded documents must include a non-empty 'vector'.")

    return len(vector)


def _build_payload(embedded_document: dict[str, Any]) -> dict[str, Any]:
    metadata = embedded_document.get("metadata", {})
    text = embedded_document.get("text") or embedded_document.get("page_content")

    if text is None:
        raise ValueError("Embedded documents must include 'text' or 'page_content'.")

    return {
        **metadata,
        "text": text,
    }


def _build_points(embedded_documents: Sequence[dict[str, Any]]) -> list[PointStruct]:
    points = []

    for point_id, embedded_document in enumerate(embedded_documents):
        points.append(
            PointStruct(
                id=embedded_document.get("id", point_id),
                vector=embedded_document["vector"],
                payload=_build_payload(embedded_document),
            )
        )

    return points


def create_collection(
    client: QdrantClient,
    embedded_documents: Sequence[dict[str, Any]],
    collection_name: str = COLLECTION_NAME,
) -> None:
    if client.collection_exists(collection_name=collection_name):
        client.delete_collection(collection_name=collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=_vector_size(embedded_documents),
            distance=Distance.COSINE,
        ),
    )


def store_vectors(
    embedded_documents: Sequence[dict[str, Any]],
    collection_name: str = COLLECTION_NAME,
    client: QdrantClient | None = None,
    batch_size: int = 100,
) -> None:
    qdrant_client = client or get_qdrant_client()
    create_collection(qdrant_client, embedded_documents, collection_name)

    points = _build_points(embedded_documents)
    for start in range(0, len(points), batch_size):
        qdrant_client.upsert(
            collection_name=collection_name,
            points=points[start : start + batch_size],
        )


def build_vectorstore(
    embedded_documents: Sequence[dict[str, Any]],
    collection_name: str = COLLECTION_NAME,
    client: QdrantClient | None = None,
    batch_size: int = 100,
) -> None:
    store_vectors(
        embedded_documents=embedded_documents,
        collection_name=collection_name,
        client=client,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    print(
        "Use build_vectorstore(embedded_documents) after embedding chunks. "
        "Expected input: {'vector': [...], 'metadata': {...}, 'text': '...'}"
    )
