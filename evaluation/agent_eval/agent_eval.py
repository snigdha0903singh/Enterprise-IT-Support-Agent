import json
from pathlib import Path

from agent.workflow import run
from retrieval.retriever import get_retriever
from tools.tools import TOOL_DEFINITIONS,TOOLS


DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "dataset"
    / "agent_benchmarking"
    / "ground_truth_dataset.json"
)


with DATA_PATH.open("r", encoding="utf-8") as file:
    data = json.load(file)


class AgentEvaluator:

    def __init__(self, dataset=None):
        self.data = dataset or data
        self.retriever = get_retriever()
    def run_agent(self, query: str):

        return run(
            query=query,
            retriever=self.retriever,
            tools=TOOLS,
            tool_descriptions=TOOL_DEFINITIONS,
        )

    ###########################################################
    # Tool Accuracy
    ###########################################################

    def get_tool_accuracy(self):

        correct = 0

        for item in self.data:

            result = self.run_agent(item["query"])

            predicted_tool = result.decision.tool
            expected_tool = item["tool_used"][0]

            if predicted_tool == expected_tool:
                correct += 1

            print("=" * 80)
            print("Query:", item["query"])
            print("Expected Tool :", expected_tool)
            print("Predicted Tool:", predicted_tool)
            print()

        return correct / len(self.data) * 100

    ###########################################################
    # Argument Accuracy
    ###########################################################

    def get_argument_accuracy(self):

        total = 0
        correct = 0

        for item in self.data:

            result = self.run_agent(item["query"])

            expected_args = item["tool_arguments"]
            predicted_args = result.arguments

            for key, expected_value in expected_args.items():

                total += 1

                if (
                    key in predicted_args
                    and predicted_args[key] == expected_value
                ):
                    correct += 1

            print("=" * 80)
            print("Query:", item["query"])
            print("Expected Args :", expected_args)
            print("Predicted Args:", predicted_args)
            print()

        if total == 0:
            return 0.0

        return correct / total * 100

    ###########################################################
    # End-to-End Success
    ###########################################################

    def get_execution_success(self):

        success = 0

        for item in self.data:

            result = self.run_agent(item["query"])

            tool_correct = (
                result.tool == item["tool"]
            )

            arguments_correct = (
                result.arguments == item["tool_arguments"]
            )

            if tool_correct and arguments_correct:
                success += 1

        return success / len(self.data) * 100

    ###########################################################
    # Evaluate
    ###########################################################

    def evaluate(self):

        tool_accuracy = self.get_tool_accuracy()
        argument_accuracy = self.get_argument_accuracy()
        execution_success = self.get_execution_success()

        print("\n" + "=" * 80)
        print(f"Tool Accuracy      : {tool_accuracy:.2f}%")
        print(f"Argument Accuracy  : {argument_accuracy:.2f}%")
        print(f"Execution Success  : {execution_success:.2f}%")
        print("=" * 80)

    ###########################################################

    def close(self):
        self.retriever.client.close()


if __name__ == "__main__":

    evaluator = AgentEvaluator()

    try:
        evaluator.evaluate()
    finally:
        evaluator.close()