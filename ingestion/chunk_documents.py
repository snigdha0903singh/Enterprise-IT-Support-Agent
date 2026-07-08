try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter


DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 300


def chunk_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n\n", "\n", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    for chunk_index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = chunk_index

    return chunks


if __name__ == "__main__":
    from ingestion.load_documents import load_all_documents
    from ingestion.metadata_builder import add_metadata_to_documents

    docs = load_all_documents()
    docs_with_metadata = add_metadata_to_documents(docs)
    chunks = chunk_documents(docs_with_metadata)

    print(f"Created {len(chunks)} chunks from {len(docs)} documents")
    print(chunks[0].metadata)
