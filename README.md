##Enterprise IT Support Agent##
Overview

Enterprise IT Support Agent is an Agentic Retrieval-Augmented Generation (RAG) system designed to answer enterprise IT support requests using organization-specific knowledge while autonomously selecting and executing appropriate enterprise tools.

The project is built to evaluate how different query transformation strategies influence both:

Retrieval quality
Downstream agent decision making

Instead of comparing different retrieval architectures, the project keeps the retrieval pipeline, reranking strategy, planner, tools, and evaluation framework identical while varying only the query transformation technique.

Problem Statement

Traditional enterprise chatbots often struggle with:

retrieving the correct document
handling ambiguous enterprise terminology
understanding policy identifiers
selecting the correct enterprise action

This project investigates whether improving the retrieval query itself leads to better downstream agent performance.

The system therefore evaluates four RAG pipelines:

Vanilla RAG
HyDE RAG
Query Rewrite RAG
Multi-Query RAG
Repository Structure
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
│
├── evaluation/
│
├── vectorstore/
│
└── scripts/
Project Workflow
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
Query Transformation Strategies

The project evaluates four retrieval pipelines.

1. Vanilla RAG

Uses the original user query directly.

Example

Give Priya Nair access to Polaris Reporting.

The original query is embedded using the BGE retrieval instruction.

2. HyDE RAG

Uses an LLM to generate a semantic search representation of the user's request.

The generated semantic passage is embedded instead of the original query.

Example

Employee Priya Nair requesting access to Polaris Reporting.
Relevant concepts include authorization,
enterprise permissions,
access policies,
security policies,
identity management,
knowledge base procedures.
3. Query Rewrite

Uses an LLM to rewrite the user query into a clearer enterprise search query.

Example

Request access for Priya Nair to the Polaris Reporting system.
4. Multi Query

Generates multiple retrieval queries representing different retrieval perspectives.

Example

Request access for Priya Nair to Polaris Reporting.

Grant enterprise access to Polaris Reporting.

Find enterprise access policies for Polaris Reporting.

Retrieve documentation describing Polaris Reporting access procedures.

Each query is independently embedded and retrieved.

The retriever merges the candidates before reranking.

Retrieval Pipeline

The retrieval architecture remains identical across all pipelines.

Embedding Model
BAAI/bge-large-en-v1.5

Query embeddings use the BGE retrieval instruction.

Vector Database
Qdrant
Retrieval

Top 50 vector candidates are retrieved.

Hybrid Reranking

Each candidate receives an additional lexical score based on:

enterprise identifiers
metadata matches
keyword overlap

The final ranking score is

Vector Similarity
+
Lexical Score

This improves retrieval for enterprise entities such as

EMP IDs
Project IDs
KB IDs
Policy IDs
Incident IDs
Agent Workflow

The retrieved context is passed to the planning agent.

The planner reasons over

user request
retrieved context
available tools

The planner returns

{
    "tool": "...",
    "arguments": {},
    "confidence": 0.95
}

The selected tool is then executed.

Enterprise Tools

The project currently simulates enterprise operations such as

Access Requests
Incident Management
Project Operations
Knowledge Base Lookup
Policy Lookup

The tool registry is intentionally modular to allow additional enterprise tools to be added without changing the planner.

Evaluation

The project evaluates both retrieval quality and agent performance.

Retrieval Metrics
Recall@1
Recall@5
Mean Reciprocal Rank (MRR)
Agent Metrics
Tool Selection Accuracy
Argument Extraction Accuracy
Execution Success Rate
Hallucination Rate
Experimental Design

Only the query transformation strategy changes between experiments.

The following components remain identical across all pipelines:

Enterprise Dataset
Chunking Strategy
Metadata Extraction
Embedding Model
Vector Store
Hybrid Reranker
Planner Agent
Tool Registry
Evaluation Dataset

This enables a controlled comparison of the effect of query transformation on retrieval and agent performance.

Available Branches

The repository currently maintains separate branches for each query transformation strategy to simplify experimentation.

Branch	Description
vanilla-rag	Original user query used directly for retrieval
hyde-rag	HyDE-based semantic query generation
query-rewrite	LLM-based query rewriting
multi-query	Multiple retrieval queries generated from a single user query

The implementation across these branches is intentionally identical except for the query transformation module imported by the retriever. Once experimentation is complete, the branches will be merged into a unified implementation where the query transformation strategy can be selected through configuration.

Future Work

Potential future enhancements include:

Reciprocal Rank Fusion (RRF) for Multi-Query retrieval
BM25 integration
Cross-encoder reranking
LangSmith experiment tracking
Tool execution against real enterprise services
Multi-agent planning
Online evaluation using user feedback
Hybrid sparse+dense retrieval
Technologies Used
Retrieval
Qdrant
HuggingFace Embeddings
BGE Large v1.5
LLMs
Qwen 3 32B (Planner)
GPT-4o Mini (Query Transformation)
Framework
Python
LangChain Document API
Evaluation
Recall@1
Recall@5
MRR
Custom Agent Evaluation Framework
