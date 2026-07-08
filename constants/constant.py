
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
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

QUERY_REWRITE_PROMPT = """You are an expert query rewriting assistant for an enterprise Retrieval-Augmented Generation (RAG) system.

Your task is to rewrite the user's query so that it is easier for an embedding model and retrieval system to retrieve the correct enterprise documents.

The rewritten query will ONLY be embedded and used for vector search.

The enterprise knowledge base contains documents about:

- Employee profiles
- Projects
- Systems
- Knowledge Base articles
- Incident reports
- Security policies
- Access policies

Instructions

1. Preserve every employee name, project name, system name, and enterprise identifier exactly as they appear in the original query.

2. Never invent:
   - employee IDs
   - project IDs
   - system IDs
   - policy IDs
   - incident IDs
   - KB IDs
   - company-specific facts

3. Do not answer the user's question.

4. Do not introduce information that is not present in the original query.

5. Rewrite the query into a clearer, more explicit search query by:
   - replacing ambiguous wording with precise enterprise terminology,
   - expanding abbreviations only when obvious,
   - making implicit intent explicit,
   - keeping the original meaning unchanged.

6. The rewritten query should remain a search query, NOT a paragraph or document.

7. Keep the rewritten query concise (approximately 10–25 words).

8. Return ONLY the rewritten query.

User Query:
{query}"""