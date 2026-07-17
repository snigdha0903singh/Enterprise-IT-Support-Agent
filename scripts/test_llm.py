
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from llm.wrapper import OpenRouterWrapper

from dotenv import load_dotenv
import os
load_dotenv()
model=os.getenv("HYDE_MODEL")
tokens=int(os.getenv("HYDE_MAX_TOKENS", "256"))
base_url=os.getenv("LOCAL_MODEL_BASE_URL")
api_key=os.getenv("LOCAL_MODEL_API_KEY")
llm = OpenRouterWrapper(model=model, tokens=tokens, api_key=api_key,base_url=base_url)

response = llm.invoke(
    "Say hello in exactly three words."
)

print(response)