import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

from ingestion.build_vectorstore import COLLECTION_NAME
from ingestion.embed_documents import DEFAULT_EMBEDDING_MODEL


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QDRANT_PATH = REPO_ROOT / "vectorstore" / "qdrant"
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=DEFAULT_EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def get_qdrant_client(
    url: str | None = None,
    path: str | Path | None = DEFAULT_QDRANT_PATH,
) -> QdrantClient:
    qdrant_url = url or os.getenv("QDRANT_URL")
    qdrant_path = path or os.getenv("QDRANT_PATH")

    if qdrant_url:
        return QdrantClient(url=qdrant_url)

    return QdrantClient(path=str(qdrant_path))


class SimpleQdrantRetriever:
    def __init__(
        self,
        client: QdrantClient,
        embeddings: HuggingFaceEmbeddings,
        collection_name: str = COLLECTION_NAME,
        k: int = 5,
        query_instruction: str = BGE_QUERY_INSTRUCTION,
    ) -> None:
        self.client = client
        self.embeddings = embeddings
        self.collection_name = collection_name
        self.k = k
        self.query_instruction = query_instruction

    def invoke(self, query: str) -> list[Document]:
        embedded_query = f"{self.query_instruction}{query}" if self.query_instruction else query
        query_vector = self.embeddings.embed_query(embedded_query)
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=self.k,
            with_payload=True,
            with_vectors=False,
        )

        documents = []
        for point in results.points:
            payload = point.payload or {}
            text = payload.get("text", "")
            metadata = {key: value for key, value in payload.items() if key != "text"}
            metadata["_id"] = point.id
            metadata["_score"] = point.score

            documents.append(
                Document(
                    page_content=text,
                    metadata=metadata,
                )
            )

        return documents


def get_retriever(
    collection_name: str = COLLECTION_NAME,
    k: int = 5,
    url: str | None = None,
    path: str | Path | None = DEFAULT_QDRANT_PATH,
):
    qdrant_client = get_qdrant_client(url=url, path=path)
    count = qdrant_client.count(collection_name=collection_name, exact=True)
    print(f"Qdrant collection '{collection_name}' contains {count.count} points")

    return SimpleQdrantRetriever(
        client=qdrant_client,
        embeddings=get_embeddings(),
        collection_name=collection_name,
        k=k,
    )
