"""Minimal smoke test: evaluate one state with all three question types.

Run:  uv run python examples/smoke_test.py
Requires AI_GATEWAY_API_KEY in .env.local (loaded automatically via dotenv
or exported in the environment).
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from ai import get_model
from ai.ops import BooleanQuestion, ChoiceQuestion, ScoreQuestion, experimental_evaluate


async def main() -> None:
    api_key = os.environ.get("AI_GATEWAY_API_KEY")
    if not api_key:
        sys.exit("AI_GATEWAY_API_KEY is not set (put it in .env.local)")

    model = get_model("typesafe-ai/jev")

    result = await experimental_evaluate(
        model,
        state={
            "ticket": "I was charged twice this month and nobody answered my "
            "last three emails. Fix this today or I'm cancelling."
        },
        questions={
            "department": ChoiceQuestion(
                instructions="Which team should handle `ticket`?",
                criteria={
                    "billing": "Charges, invoices, refunds",
                    "technical": "Bugs, outages, integrations",
                    "sales": "Pricing, upgrades, new accounts",
                },
            ),
            "urgency": ScoreQuestion(
                instructions="How urgent is `ticket`?",
                criteria=["can wait", "this week", "today"],
            ),
            "is_churn_risk": BooleanQuestion(
                instructions="Does `ticket` express an intention to cancel the service?",
            ),
        },
    )

    eval_result = result.value
    for qid, answer in eval_result.answers.items():
        print(f"{qid}: {answer}")
    print(f"usage: {result.usage}")
    print(f"provider_metadata: {result.provider_metadata}")


if __name__ == "__main__":
    # Load .env.local if python-dotenv is available, else require exported var.
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[1] / ".env.local")
    except ImportError:
        pass
    asyncio.run(main())
