from pathlib import Path

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document


DATASET_DIR = Path(__file__).resolve().parents[1] / "dataset" / "documents"


def _load_markdown_docs(folder_path: Path) -> list[Document]:
    documents = []

    for file_path in sorted(folder_path.glob("*.md")):
        documents.append(
            Document(
                page_content=file_path.read_text(encoding="utf-8"),
                metadata={"source": file_path.name},
            )
        )

    return documents


def load_employee_docs() -> list[Document]:
    return _load_markdown_docs(DATASET_DIR / "employees")


def load_project_docs() -> list[Document]:
    return _load_markdown_docs(DATASET_DIR / "projects")


def load_policy_docs() -> list[Document]:
    return (
        _load_markdown_docs(DATASET_DIR / "access_policies")
        + _load_markdown_docs(DATASET_DIR / "security_policies")
    )


def load_system_docs() -> list[Document]:
    return _load_markdown_docs(DATASET_DIR / "systems")


def load_knowledge_base_docs() -> list[Document]:
    return _load_markdown_docs(DATASET_DIR / "knowledge_base")


def load_incident_guide_docs() -> list[Document]:
    return _load_markdown_docs(DATASET_DIR / "incident_guides")


def load_all_documents() -> list[Document]:
    return (
        load_employee_docs()
        + load_project_docs()
        + load_policy_docs()
        + load_system_docs()
        + load_knowledge_base_docs()
        + load_incident_guide_docs()
    )


if __name__ == "__main__":
    docs = load_all_documents()
    print(f"Loaded {len(docs)} markdown documents")
