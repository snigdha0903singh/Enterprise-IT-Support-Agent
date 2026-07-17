
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.workflow import run
from tools.tools import TOOLS, TOOL_DEFINITIONS
from retrieval.retriever import get_retriever

retriever = get_retriever()
# from llm.wrapper import OpenRouterWrapper
# llm = OpenRouterWrapper()

result = run(
    query="Give Priya Nair access to Polaris Reporting.",
    retriever=retriever,
    tools=TOOLS,
    tool_descriptions=TOOL_DEFINITIONS,
)

# print(result)
