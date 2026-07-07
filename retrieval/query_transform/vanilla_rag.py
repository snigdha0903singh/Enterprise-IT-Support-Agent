from retrieval.query_transform.model import RetrievalQuery
from constants.constant import BGE_QUERY_INSTRUCTION


class VanillaTransformer:

    def __init__(self,query_instruction: str = BGE_QUERY_INSTRUCTION):
        self.query_instruction = query_instruction

    def transform(
        self,
        query: str,
    ) -> RetrievalQuery:
        print(f"Transforming query in vanilla rag: {query}")
        search_query = f"{self.query_instruction}{query}" if self.query_instruction else query
        return RetrievalQuery(
            original_query=query,
            search_queries=[search_query],
        )