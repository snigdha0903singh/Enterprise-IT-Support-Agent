
from retrieval.query_transform.model import RetrievalQuery
from constants.constant import BGE_QUERY_INSTRUCTION,HYDE_TEXT_PROMPT
from llm.wrapper import OpenRouterWrapper
from dotenv import load_dotenv
import os
load_dotenv()
model=os.getenv("HYDE_MODEL")
tokens=int(os.getenv("HYDE_MAX_TOKENS", "256"))
base_url=os.getenv("LOCAL_MODEL_BASE_URL")
api_key=os.getenv("LOCAL_MODEL_API_KEY")
llm = OpenRouterWrapper(model=model, tokens=tokens, api_key=api_key,base_url=base_url)

class HyDETransformer:
    def __init__(self, query_instruction: str = BGE_QUERY_INSTRUCTION):
        self.query_instruction = query_instruction
        self.llm = llm

    def transform(
        self,
        query: str,
    ) -> RetrievalQuery:
        prompt = HYDE_TEXT_PROMPT.format(
            query=query,
        )

        hypothetical_document = self.llm.invoke(prompt)
        search_query = f"{self.query_instruction}{hypothetical_document}" if self.query_instruction else hypothetical_document
        print(f"Transforming query in HyDE: {query} -> {hypothetical_document}")
        return RetrievalQuery(
            original_query=query,
            search_queries=[search_query],
        )
