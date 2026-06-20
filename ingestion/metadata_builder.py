import re

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document


FIELD_PATTERN = r"\*\*{field}:\*\*\s*(.+)"


def _extract_heading_value(content: str, prefix: str) -> str | None:
    match = re.search(rf"^# {re.escape(prefix)}:\s*(.+)$", content, re.MULTILINE)
    return match.group(1).strip() if match else None


def _extract_field(content: str, field: str) -> str | None:
    pattern = FIELD_PATTERN.format(field=re.escape(field))
    match = re.search(pattern, content)
    return match.group(1).strip() if match else None


def _source(document: Document) -> str | None:
    return document.metadata.get("source")


def build_employee_metadata(document: Document) -> dict:
    content = document.page_content

    return {
        "doc_type": "employee",
        "employee_id": _extract_field(content, "Employee ID"),
        "employee_name": _extract_heading_value(content, "Employee Profile"),
        "department": _extract_field(content, "Department"),
        "business_unit": _extract_field(content, "Business Unit"),
        "title": _extract_field(content, "Title"),
        "region": _extract_field(content, "Region"),
        "source": _source(document),
    }


def build_project_metadata(document: Document) -> dict:
    content = document.page_content

    return {
        "doc_type": "project",
        "project_id": _extract_field(content, "Project ID"),
        "project_name": _extract_heading_value(content, "Project Documentation"),
        "business_unit": _extract_field(content, "Business Unit"),
        "status": _extract_field(content, "Status"),
        "security_classification": _extract_field(
            content, "Security Classification"
        ),
        "source": _source(document),
    }


def build_policy_metadata(document: Document) -> dict:
    content = document.page_content
    source = _source(document) or ""

    if source.startswith("security_"):
        policy_type = "security_policy"
        policy_name = _extract_heading_value(content, "Security Policy")
    else:
        policy_type = "access_policy"
        policy_name = _extract_heading_value(content, "Access Policy")

    return {
        "doc_type": "policy",
        "policy_type": policy_type,
        "policy_id": _extract_field(content, "Policy ID"),
        "policy_name": policy_name,
        "system_covered": _extract_field(content, "System Covered"),
        "severity_threshold": _extract_field(content, "Severity Threshold"),
        "source": source,
    }


def build_metadata(document: Document) -> dict:
    source = _source(document) or ""

    if source.startswith("employee_"):
        return build_employee_metadata(document)
    if source.startswith("project_"):
        return build_project_metadata(document)
    if source.startswith(("policy_", "security_")):
        return build_policy_metadata(document)

    return {
        "doc_type": "unknown",
        "source": source,
    }


def add_metadata(document: Document) -> Document:
    metadata = {
        **document.metadata,
        **build_metadata(document),
    }

    return Document(
        page_content=document.page_content,
        metadata={key: value for key, value in metadata.items() if value is not None},
    )


def add_metadata_to_documents(documents: list[Document]) -> list[Document]:
    return [add_metadata(document) for document in documents]
