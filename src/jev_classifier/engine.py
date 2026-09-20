"""Jev classification engine via the Vercel AI Gateway.

The only supported classification backend for this project. Uses the
`ai` Python SDK's ``experimental_evaluate`` with model ``typesafe-ai/jev``,
which evaluates typed questions (choice / score / boolean) against a shared
state and returns typed answers with probabilities.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from ai import get_model
from ai.ops import BooleanQuestion, ChoiceQuestion, ScoreQuestion, experimental_evaluate

MODEL_ID = "typesafe-ai/jev"


@dataclass(frozen=True)
class Label:
    """A classification label: the option key and its rubric description."""

    name: str
    description: str = ""


@dataclass
class ClassificationResult:
    """Typed result of classifying one text against one label set."""

    label: str
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float | None = None
    urgency: float | None = None  # fractional score across the urgency rubric
    is_complaint: bool | None = None
    complaint_probability: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)


def _parse_usage(item: Any) -> tuple[int | None, int | None]:
    usage = getattr(item, "usage", None)
    if usage is None:
        return None, None
    return (
        getattr(usage, "input_tokens", None),
        getattr(usage, "output_tokens", None),
    )


async def classify_async(
    text: str,
    labels: list[Label],
    *,
    ask_urgency: bool = True,
    ask_complaint: bool = True,
    model_id: str = MODEL_ID,
) -> ClassificationResult:
    """Classify one text. All questions run in parallel in one request."""
    if not labels:
        raise ValueError("labels must not be empty")
    if len(labels) > 255:
        raise ValueError("Jev supports at most 255 options per Choice question")

    questions: dict[str, Any] = {
        "category": ChoiceQuestion(
            instructions="Which category does `text` belong to?",
            criteria={lb.name: (lb.description or None) for lb in labels},
        ),
    }
    if ask_urgency:
        questions["urgency"] = ScoreQuestion(
            instructions="How urgent is `text`?",
            criteria=["not urgent, can wait", "should be handled this week", "urgent, handle today"],
        )
    if ask_complaint:
        questions["is_complaint"] = BooleanQuestion(
            instructions="Does `text` express a complaint or dissatisfaction?",
        )

    item: Any = None
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            item = await experimental_evaluate(
                get_model(model_id),
                state={"text": text},
                questions=questions,
            )
            break
        except Exception as exc:  # transient gateway timeouts/5xx: retry with backoff
            last_exc = exc
            if attempt == 2:
                raise
            await asyncio.sleep(2**attempt * 2)
    assert item is not None, f"evaluate failed: {last_exc}"

    answers = item.value.answers
    category = answers["category"]
    input_tokens, output_tokens = _parse_usage(item)

    # TypeSafe exposes its own confidence statistic via providerMetadata,
    # keyed by question id; fall back to the selected option's probability.
    ts_meta = (item.provider_metadata or {}).get("typesafe") or {}
    confidence = ts_meta.get("confidence", {}).get("category")
    if confidence is None and category.probabilities:
        confidence = category.probabilities.get(category.choice)

    result = ClassificationResult(
        label=category.choice,
        probabilities=dict(category.probabilities or {}),
        confidence=confidence,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        raw=item.model_dump(mode="json"),
    )
    if ask_urgency:
        result.urgency = answers["urgency"].score
    if ask_complaint:
        result.is_complaint = answers["is_complaint"].probability >= 0.5
        result.complaint_probability = answers["is_complaint"].probability
    return result


def classify(
    text: str,
    labels: list[Label],
    *,
    ask_urgency: bool = True,
    ask_complaint: bool = True,
    model_id: str = MODEL_ID,
) -> ClassificationResult:
    """Synchronous wrapper around :func:`classify_async`."""
    return asyncio.run(
        classify_async(text, labels, ask_urgency=ask_urgency, ask_complaint=ask_complaint, model_id=model_id)
    )
