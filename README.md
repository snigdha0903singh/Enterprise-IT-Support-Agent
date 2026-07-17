# Enterprise IT Support Agent

An **Agentic Retrieval-Augmented Generation (RAG)** system that answers enterprise IT support requests using organization-specific knowledge, while autonomously selecting and executing the appropriate enterprise tools.

---

## 📌 Overview

This project investigates how different **query transformation strategies** affect:

- **Retrieval quality** — how well the right documents are found
- **Downstream agent decision-making** — how well the agent selects and executes the correct tool

Rather than comparing different retrieval *architectures*, the project holds the retrieval pipeline, reranking strategy, planner, tools, and evaluation framework **constant**, varying only the **query transformation technique**.

---

## ❓ Problem Statement

Traditional enterprise chatbots often struggle with:

- Retrieving the correct document
- Handling ambiguous enterprise terminology
- Understanding policy identifiers
- Selecting the correct enterprise action

This project investigates whether **improving the retrieval query itself** leads to better downstream agent performance.

Four RAG pipelines are evaluated:

| Pipeline | Description |
|---|---|
| **Vanilla RAG** | Uses the original user query directly |
| **HyDE RAG** | Uses an LLM-generated hypothetical passage for retrieval |
| **Query Rewrite RAG** | Uses an LLM to rewrite the query into a clearer enterprise search query |
| **Multi-Query RAG** | Generates multiple queries representing different retrieval perspectives |

---

## 🗂️ Repository Structure

```
Enterprise-IT-Support-Agent
│
├── agent/
│   ├── planner.py
│   ├── workflow.py
│   ├── state.py
│   ├── prompts.py
│   ├── tool_executor.py
│   └── ...
│
├── dataset/
│   ├── documents/
│   ├── generation_scripts/
│   └── evaluation/
│
├── ingestion/
│   ├── load_documents.py
│   ├── metadata_builder.py
│   ├── embed_documents.py
│   ├── chunk_documents.py
│   └── build_vectorstore.py
│
├── llm/
│   └── wrapper.py
│
├── retrieval/
│   ├── retriever.py
│   └── query_transform/
│       ├── vanilla_rag.py
│       ├── hyde.py
│       ├── query_rewrite.py
│       ├── multi_query.py
│       └── model.py
│
├── tools/
├── evaluation/
├── vectorstore/
└── scripts/
```

---

## 🔄 Project Workflow

```
User Query
      │
      ▼
Query Transformation
(Vanilla / HyDE / Query Rewrite / Multi Query)
      │
      ▼
Embedding Generation (BGE)
      │
      ▼
Dense Vector Retrieval (Qdrant)
      │
      ▼
Hybrid Lexical Reranking
      │
      ▼
Retrieved Enterprise Context
      │
      ▼
Planner Agent
      │
      ▼
Tool Selection
      │
      ▼
Tool Execution
      │
      ▼
Final Response
```

---

## 🔍 Query Transformation Strategies

### 1. Vanilla RAG
Uses the original user query directly, embedded with the BGE retrieval instruction.

**Example**
```
Give Priya Nair access to Polaris Reporting.
```

### 2. HyDE RAG
Uses an LLM to generate a semantic search representation ("hypothetical document") of the user's request. The generated passage is embedded instead of the original query.

**Example**
```
Employee Priya Nair requesting access to Polaris Reporting.
Relevant concepts include authorization, enterprise permissions,
access policies, security policies, identity management,
knowledge base procedures.
```

### 3. Query Rewrite
Uses an LLM to rewrite the user query into a clearer enterprise search query.

**Example**
```
Request access for Priya Nair to the Polaris Reporting system.
```

### 4. Multi-Query
Generates multiple retrieval queries representing different retrieval perspectives. Each is independently embedded and retrieved, then merged before reranking.

**Example**
```
Request access for Priya Nair to Polaris Reporting.
Grant enterprise access to Polaris Reporting.
Find enterprise access policies for Polaris Reporting.
Retrieve documentation describing Polaris Reporting access procedures.
```

---

## 🧱 Retrieval Pipeline

The retrieval architecture remains **identical** across all pipelines.

| Component | Details |
|---|---|
| **Embedding Model** | `BAAI/bge-large-en-v1.5` (BGE retrieval instruction used for query embeddings) |
| **Vector Database** | Qdrant |
| **Retrieval** | Top 50 vector candidates retrieved |
| **Hybrid Reranking** | Lexical score added on top of vector similarity |

**Final ranking score:**

```
Final Score = Vector Similarity + Lexical Score
```

The lexical score is based on:
- Enterprise identifiers
- Metadata matches
- Keyword overlap

This improves retrieval for enterprise entities such as:
- EMP IDs
- Project IDs
- KB IDs
- Policy IDs
- Incident IDs

---

## 🤖 Agent Workflow

The retrieved context is passed to the **planning agent**, which reasons over:

- The user request
- The retrieved context
- Available tools

The planner returns a structured decision:

```json
{
    "tool": "...",
    "arguments": {},
    "confidence": 0.95
}
```

The selected tool is then executed.

---

## 🛠️ Enterprise Tools

The project currently simulates enterprise operations such as:

- Access Requests
- Incident Management
- Project Operations
- Knowledge Base Lookup
- Policy Lookup

The tool registry is intentionally modular, allowing new enterprise tools to be added without changing the planner.

---

## 📊 Evaluation

The project evaluates both **retrieval quality** and **agent performance**.

**Retrieval Metrics**
- Recall@1
- Recall@5
- Mean Reciprocal Rank (MRR)

**Agent Metrics**
- Tool Selection Accuracy
- Argument Extraction Accuracy
- Execution Success Rate
- Hallucination Rate

---

## 🧪 Experimental Design

Only the **query transformation strategy** changes between experiments. The following components remain identical across all pipelines:

- Enterprise Dataset
- Chunking Strategy
- Metadata Extraction
- Embedding Model
- Vector Store
- Hybrid Reranker
- Planner Agent
- Tool Registry
- Evaluation Dataset

This enables a **controlled comparison** of the effect of query transformation on retrieval and agent performance.

---

## 🌿 Available Branches

The repository maintains separate branches for each query transformation strategy to simplify experimentation.

| Branch | Description |
|---|---|
| `vanilla-rag` | Original user query used directly for retrieval |
| `hyde-rag` | HyDE-based semantic query generation |
| `query-rewrite` | LLM-based query rewriting |
| `multi-query` | Multiple retrieval queries generated from a single user query |

The implementation across these branches is intentionally identical except for the query transformation module imported by the retriever. Once experimentation is complete, the branches will be merged into a unified implementation where the strategy can be selected via configuration.

---

## 🚀 Future Work

- Reciprocal Rank Fusion (RRF) for Multi-Query retrieval
- BM25 integration
- Cross-encoder reranking
- LangSmith experiment tracking
- Tool execution against real enterprise services
- Multi-agent planning
- Online evaluation using user feedback
- Hybrid sparse + dense retrieval

---

## 🧰 Technologies Used

**Retrieval**
- Qdrant
- HuggingFace Embeddings
- BGE Large v1.5

**LLMs**
- Qwen 3 32B (Planner)
- GPT-4o Mini (Query Transformation)

**Framework**
- Python
- LangChain Document API

**Evaluation**
- Recall@1, Recall@5, MRR
- Custom Agent Evaluation Framework