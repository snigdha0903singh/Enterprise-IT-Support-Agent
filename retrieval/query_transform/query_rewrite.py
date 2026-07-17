from retrieval.query_transform.model import RetrievalQuery
from constants.constant import BGE_QUERY_INSTRUCTION,QUERY_REWRITE_PROMPT
from llm.wrapper import OpenRouterWrapper
from dotenv import load_dotenv
import os
load_dotenv()
model=os.getenv("HYDE_MODEL")
tokens=int(os.getenv("HYDE_MAX_TOKENS", "256"))
base_url=os.getenv("LOCAL_MODEL_BASE_URL")
api_key=os.getenv("LOCAL_MODEL_API_KEY")
llm = OpenRouterWrapper(model=model, tokens=tokens, api_key=api_key,base_url=base_url)

class QueryRewriter:
    def __init__(self, query_instruction: str = BGE_QUERY_INSTRUCTION):
        self.query_instruction = query_instruction
        self.llm = llm

    def transform(
        self,
        query: str,
    ) -> RetrievalQuery:
        prompt = QUERY_REWRITE_PROMPT.format(
            query=query,
        )

        reformed_query = self.llm.invoke(prompt)
        search_query = f"{self.query_instruction}{reformed_query}" if self.query_instruction else reformed_query
        print(f"Transforming query in QueryRewriter: {query} -> {reformed_query}")
        return RetrievalQuery(
            original_query=query,
            search_queries=[search_query],
        )