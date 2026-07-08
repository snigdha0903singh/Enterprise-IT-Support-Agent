from retrieval.query_transform.model import RetrievalQuery
from constants.constant import BGE_QUERY_INSTRUCTION,MULTI_QUERY_PROMPT
from llm.wrapper import OpenRouterWrapper
from dotenv import load_dotenv
import os
load_dotenv()
model=os.getenv("HYDE_MODEL")
tokens=int(os.getenv("HYDE_MAX_TOKENS", "256"))
llm = OpenRouterWrapper(model=model, tokens=tokens)

class MultiQueryTransformer:
    def __init__(self, query_instruction: str = BGE_QUERY_INSTRUCTION):
        self.query_instruction = query_instruction
        self.llm = llm

    def transform(
        self,
        query: str,
    ) -> RetrievalQuery:
        prompt = MULTI_QUERY_PROMPT.format(
            query=query,
        )

        reformed_query = self.llm.invoke(prompt)
        queries = [
            line.strip()
            for line in reformed_query.splitlines()
            if line.strip()
        ]
        queries.append(query)
        print(f"Transforming query in Multi query: {query} -> {queries}")
        return RetrievalQuery(
            original_query=query,
            search_queries=queries,
        )