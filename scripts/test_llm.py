
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from llm.wrapper import OpenRouterWrapper

llm = OpenRouterWrapper()

response = llm.invoke(
    "Say hello in exactly three words."
)

print(response)