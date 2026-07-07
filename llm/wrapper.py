from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class OpenRouterWrapper:
    """
    Wrapper around OpenRouter that satisfies the PlannerLLM protocol.

    The planner only expects:

        llm.invoke(prompt)

    It should not know anything about API keys,
    endpoints, providers, or models.
    """

    def __init__(
        self,
        model: str |None = None,
        temperature: float | None = None,
        tokens: int | None = None,
    ) -> None:

        api_key = os.getenv("OPENROUTER_API_KEY")

        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not found in .env"
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url=os.getenv(
                "OPENROUTER_BASE_URL",
                "https://openrouter.ai/api/v1",
            ),
        )

        self.model = (
            model
            or os.getenv(
                "OPENROUTER_MODEL",
                "openai/gpt-4o",
            )
        )

        self.temperature = (
            temperature
            if temperature is not None
            else float(os.getenv("TEMPERATURE", "0.0"))
        )

        self.tokens = (tokens if tokens is not None else int(os.getenv("MAX_TOKENS", "512")))

    def invoke(self, prompt: str) -> str:
        """
        Send a prompt to the configured model.

        Returns only the generated text.
        """

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.tokens,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an enterprise IT planning agent. "
                        "Always follow instructions carefully and "
                        "return only what is requested."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        message = response.choices[0].message
        # print(f"Model response complete: {message}")
        if message.content is None:
             raise ValueError(
                  "Model returned no content."
                  )
        return message.content
