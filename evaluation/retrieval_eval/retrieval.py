import json
from pathlib import Path

from retrieval.retriever import get_retriever

DATA_PATH = Path(__file__).resolve().parents[1] / "dataset" / "retrieval_benchmarking" / "data.json"

with DATA_PATH.open("r", encoding="utf-8") as file:
    data = json.load(file)

class Evaluators:
    def __init__(self, data):
        self.data = data
        self.retriever = get_retriever()

    def get_retrieved_doc_sources(self, query: str) -> tuple[list[str], dict[str, int]]:
        docs = self.retriever.invoke(query)
        retrieved_docs = []
        doc_rank_by_source = {}

        for rank, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source")
            if not source:
                continue

            if source in doc_rank_by_source:
                continue

            retrieved_docs.append(source)
            doc_rank_by_source[source] = rank

        return retrieved_docs, doc_rank_by_source

    def get_unique_retrieved_doc_list(self, query: str):
        retrieved_docs, _ = self.get_retrieved_doc_sources(query)
        return retrieved_docs

    def get_unique_retrievied_doc_list(self, query: str):
        return self.get_unique_retrieved_doc_list(query)

    def get_relevant_docs(self, item: dict) -> list:
        return item.get("ground_truth") or item.get("relevant_docs") or []

    def get_recall_at_5_score(self, retrieved_docs: list, relevant_docs: list):
        if not relevant_docs:
            return 0.0

        top_5_docs = retrieved_docs[:5]
        retrieved_relevant_docs = [doc for doc in top_5_docs if doc in relevant_docs]
        recall_at_5 = len(retrieved_relevant_docs) / len(relevant_docs)
        return recall_at_5

    def get_recall_percentage(self):
        recall_score = 0
        for item in self.data:
            query = item["query"]
            relevant_docs = self.get_relevant_docs(item)
            retrieved_docs = self.get_unique_retrieved_doc_list(query)
            recall_at_5 = self.get_recall_at_5_score(retrieved_docs, relevant_docs)
            recall_score += recall_at_5
            print(f"Query: {query}")
            print(f"Relevant Docs: {relevant_docs}")
            print(f"Retrieved Docs: {retrieved_docs}")
            print(f"Recall@5: {recall_at_5:.2f}\n")
        recall_percentage = recall_score / len(self.data) * 100
        return recall_percentage
    
    def get_mrr_score(self, doc_rank_by_source: dict[str, int], relevant_docs: list):
        if not relevant_docs:
            return 0.0

        relevant_ranks = [
            doc_rank_by_source[doc]
            for doc in relevant_docs
            if doc in doc_rank_by_source
        ]
        if relevant_ranks:
            return 1 / min(relevant_ranks)

        return 0.0

    def get_mrr_percentage(self):
        mrr_score = 0
        for item in self.data:
            query = item["query"]
            relevant_docs = self.get_relevant_docs(item)
            retrieved_docs, doc_rank_by_source = self.get_retrieved_doc_sources(query)
            reciprocal_rank = self.get_mrr_score(doc_rank_by_source, relevant_docs)
            mrr_score += reciprocal_rank
            print(f"Query: {query}")
            print(f"Relevant Docs: {relevant_docs}")
            print(f"Retrieved Docs: {retrieved_docs}")
            print(f"Doc Ranks: {doc_rank_by_source}")
            print(f"MRR: {reciprocal_rank:.2f}\n")

        return mrr_score / len(self.data) * 100
    
