import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.agent_eval.agent_eval import AgentEvaluator


if __name__ == "__main__":

    evaluator = AgentEvaluator()

    try:
        evaluator.evaluate()

    finally:
        evaluator.close()