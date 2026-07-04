import os
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

from ingestion.build_vectorstore import COLLECTION_NAME
from ingestion.embed_documents import DEFAULT_EMBEDDING_MODEL
from ingestion.load_documents import load_all_documents
from ingestion.metadata_builder import add_metadata_to_documents


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QDRANT_PATH = REPO_ROOT / "vectorstore" / "qdrant"
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
IDENTIFIER_PATTERN = re.compile(r"\b(?:ACP|SEC|EMP|KB|INC|SYS|PRJ)-\d{4,5}\b", re.IGNORECASE)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "article",
    "by",
    "does",
    "for",
    "guide",
    "how",
    "in",
    "is",
    "of",
    "on",
    "policy",
    "project",
    "system",
    "the",
    "to",
    "under",
    "what",
    "which",
    "who",
}
NAME_METADATA_FIELDS = (
    "employee_name",
    "project_name",
    "policy_name",
    "system_name",
    "article_name",
    "guide_name",
)


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
        candidate_k: int = 50,
        expand_source_documents: bool = True,
        query_instruction: str = BGE_QUERY_INSTRUCTION,
    ) -> None:
        self.client = client
        self.embeddings = embeddings
        self.collection_name = collection_name
        self.k = k
        self.candidate_k = candidate_k
        self.expand_source_documents = expand_source_documents
        self.query_instruction = query_instruction
        self._payload_documents: list[Document] | None = None
        self._source_documents: dict[str, Document] | None = None

    def _document_from_point(self, point, score: float | None = None) -> Document:
        payload = point.payload or {}
        text = payload.get("text", "")
        metadata = {key: value for key, value in payload.items() if key != "text"}
        metadata["_id"] = point.id
        if score is not None:
            metadata["_score"] = score

        return Document(page_content=text, metadata=metadata)

    def _load_payload_documents(self) -> list[Document]:
        if self._payload_documents is not None:
            return self._payload_documents

        documents = []
        next_page_offset = None

        while True:
            points, next_page_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=512,
                offset=next_page_offset,
                with_payload=True,
                with_vectors=False,
            )
            documents.extend(self._document_from_point(point) for point in points)

            if next_page_offset is None:
                break

        self._payload_documents = documents
        return documents

    def _load_source_documents(self) -> dict[str, Document]:
        if self._source_documents is not None:
            return self._source_documents

        documents = add_metadata_to_documents(load_all_documents())
        self._source_documents = {
            document.metadata["source"]: document
            for document in documents
            if document.metadata.get("source")
        }
        return self._source_documents

    def _expand_to_source_document(self, document: Document) -> Document:
        source = document.metadata.get("source")
        if not source:
            return document

        source_document = self._load_source_documents().get(source)
        if source_document is None:
            return document

        return Document(
            page_content=source_document.page_content,
            metadata={
                **source_document.metadata,
                "_retrieved_chunk": document.page_content,
                "_retrieved_chunk_index": document.metadata.get("chunk_index"),
                "_score": document.metadata.get("_score"),
            },
        )

    def _lexical_score(self, query: str, document: Document) -> float:
        query_lower = query.lower()
        metadata = document.metadata
        searchable_parts = [
            str(value)
            for key, value in metadata.items()
            if not key.startswith("_") and value is not None
        ]
        searchable_parts.append(document.page_content)
        searchable_text = "\n".join(searchable_parts).lower()

        score = 0.0
        for identifier in IDENTIFIER_PATTERN.findall(query):
            if identifier.lower() in searchable_text:
                score += 20.0

        for field in NAME_METADATA_FIELDS:
            value = metadata.get(field)
            if value and str(value).lower() in query_lower:
                score += 8.0

        query_tokens = [
            token
            for token in TOKEN_PATTERN.findall(query_lower)
            if token not in STOPWORDS and len(token) > 1
        ]
        if query_tokens:
            searchable_tokens = set(TOKEN_PATTERN.findall(searchable_text))
            overlap = sum(1 for token in query_tokens if token in searchable_tokens)
            score += overlap / len(query_tokens)

        return score

    def invoke(self, query: str) -> list[Document]:
        embedded_query = f"{self.query_instruction}{query}" if self.query_instruction else query
        query_vector = self.embeddings.embed_query(embedded_query)
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=max(self.k, self.candidate_k),
            with_payload=True,
            with_vectors=False,
        )

        scored_by_source = {}
        for point in results.points:
            document = self._document_from_point(point, score=point.score)
            source = document.metadata.get("source") or document.metadata.get("_id")
            lexical_score = self._lexical_score(query, document)
            combined_score = float(point.score) + lexical_score
            existing = scored_by_source.get(source)
            if existing is None or combined_score > existing[0]:
                scored_by_source[source] = (combined_score, document)

        for document in self._load_payload_documents():
            lexical_score = self._lexical_score(query, document)
            if lexical_score <= 0:
                continue

            source = document.metadata.get("source") or document.metadata.get("_id")
            combined_score = lexical_score
            document.metadata["_score"] = combined_score
            existing = scored_by_source.get(source)
            if existing is None or combined_score > existing[0]:
                scored_by_source[source] = (combined_score, document)

        ranked_documents = sorted(
            scored_by_source.values(),
            key=lambda item: item[0],
            reverse=True,
        )
        documents = [document for _, document in ranked_documents[: self.k]]
        if not self.expand_source_documents:
            return documents

        return [self._expand_to_source_document(document) for document in documents]


def get_retriever(
    collection_name: str = COLLECTION_NAME,
    k: int = 5,
    expand_source_documents: bool = True,
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
        expand_source_documents=expand_source_documents,
    )
