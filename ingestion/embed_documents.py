from collections.abc import Sequence

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from langchain_huggingface import HuggingFaceEmbeddings


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"


def embed_documents(
    documents: Sequence[Document],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 64,
) -> list[dict]:
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        show_progress=True,
        encode_kwargs={
            "batch_size": batch_size,
            "normalize_embeddings": True,
        },
    )
    texts = [document.page_content for document in documents]
    vectors = embeddings.embed_documents(texts)

    embedded_documents = []
    for index, (document, vector) in enumerate(zip(documents, vectors)):
        embedded_documents.append(
            {
                "id": index,
                "vector": vector.tolist() if hasattr(vector, "tolist") else vector,
                "metadata": document.metadata,
                "text": document.page_content,
            }
        )

    return embedded_documents
