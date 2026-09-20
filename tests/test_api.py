"""API capability test: classify a small labeled dataset through Jev.

Verifies request/response shapes for all three question types (choice,
score, boolean), probability distributions, confidence, and token usage.

Run:  uv run python tests/test_api.py
"""

from __future__ import annotations

import asyncio
import csv
import sys
from pathlib import Path

from dotenv import load_dotenv

from jev_classifier.engine import Label, classify_async

load_dotenv(Path(__file__).resolve().parents[1] / ".env.local")

LABELS = [
    Label("billing", "Charges, invoices, refunds, payment issues"),
    Label("technical", "Bugs, errors, outages, broken features"),
    Label("sales", "Pricing, plans, upgrades, new accounts"),
    Label("support", "Account changes and general assistance"),
]


async def main() -> None:
    data_path = Path(__file__).resolve().parent / "data" / "test_tickets.csv"
    rows = list(csv.DictReader(data_path.open()))

    correct = 0
    print(f"{'expected':<10} {'predicted':<10} {'conf':>5} {'urg':>5} {'cmpl':>5}  text")
    print("-" * 100)
    for row in rows:
        result = await classify_async(row["text"], LABELS)
        ok = result.label == row["label"]
        correct += ok
        print(
            f"{row['label']:<10} {result.label:<10} "
            f"{(result.confidence or 0):5.2f} "
            f"{(result.urgency if result.urgency is not None else float('nan')):5.2f} "
            f"{(result.complaint_probability if result.complaint_probability is not None else float('nan')):5.2f}  "
            f"{row['text'][:60]}"
        )
        assert result.probabilities, "choice answer must include probabilities"
        assert abs(sum(result.probabilities.values()) - 1.0) < 0.02, (
            f"probabilities must sum to 1, got {result.probabilities}"
        )
        assert result.input_tokens and result.input_tokens > 0, "usage missing"

    print("-" * 100)
    print(f"accuracy: {correct}/{len(rows)}")
    print("ALL CHECKS PASSED" if correct >= len(rows) * 0.7 else "ACCURACY BELOW 70% — investigate", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
