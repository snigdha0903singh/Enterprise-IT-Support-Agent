from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class RetrievalQuery:

    original_query: str

    search_queries: list[str]