
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

HYDE_TEXT_PROMPT ="""You are generating semantic search text for an enterprise Retrieval-Augmented Generation (RAG) system.

The generated text will NOT be shown to the user.

It will ONLY be converted into an embedding and used for vector retrieval.

Your objective is to maximize retrieval quality by producing a semantically rich representation of the user's information need.

The enterprise knowledge base contains documents about:

- Employee profiles
- Projects
- Systems
- Knowledge Base articles
- Incident reports
- Security policies
- Access policies

Relationships commonly found include:

- Employees work on projects.
- Employees use enterprise systems.
- Projects are associated with enterprise systems.
- Systems are governed by security policies.
- Systems are governed by access policies.
- Knowledge Base articles describe enterprise procedures and troubleshooting.
- Incident reports describe issues affecting enterprise systems.
- Access policies define authorization requirements.
- Security policies define compliance requirements.

Rules

1. Preserve every employee name, project name, system name and enterprise identifier EXACTLY as they appear in the query.

2. Never invent:
   - employee IDs
   - project IDs
   - system IDs
   - KB IDs
   - incident IDs
   - policy IDs
   - project owners
   - employee roles
   - departments
   - approvals
   - security requirements
   - company-specific facts

3. If the query does not contain an identifier, DO NOT generate one.

4. Do not answer the user's question.

5. Do not speculate.

6. Expand the query only by introducing semantically related enterprise concepts that are commonly associated with the request.

7. Focus on concepts rather than storytelling.

8. Keep the output concise (approximately 40–80 words).

9. Return only the generated semantic search passage.

User Query:
{query}"""

