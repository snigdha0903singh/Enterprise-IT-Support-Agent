
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

MULTI_QUERY_PROMPT="""
You are generating multiple search queries for an enterprise Retrieval-Augmented Generation (RAG) system.

The generated queries will ONLY be embedded and used for vector retrieval.

Your goal is to maximize document recall by producing diverse search queries that express the same information need from different retrieval perspectives.

The enterprise knowledge base contains documents about:

- Employee profiles
- Projects
- Systems
- Knowledge Base articles
- Incident reports
- Security policies
- Access policies

Instructions

1. Generate exactly FOUR search queries.

2. Every query must preserve the original intent.

3. Preserve every employee name, project name, system name, and enterprise identifier exactly as they appear in the original query.

4. Never invent:
   - employee IDs
   - project IDs
   - system IDs
   - KB IDs
   - incident IDs
   - policy IDs
   - company-specific facts

5. Each query should represent a DIFFERENT retrieval perspective. For example:
   - direct request
   - enterprise terminology
   - entity-focused search
   - policy/procedure-oriented search

6. Do not answer the question.

7. Do not generate explanations.

8. Return ONLY the rewritten queries.

9. Output one query per line with no numbering or bullets.

User Query:
{query}"""

