# jev-classifier

Text classification desktop app powered by [TypeSafe Jev](https://docs.typesafe.ai)
via the [Vercel AI Gateway](https://vercel.com/ai-gateway) — a Python port of the
[ollama-classifier-gui](https://github.com/paluigi-moltis/ollama-classifier-gui)
workflow with a single, Jev-only classification engine.

## Why the `ai` package instead of `typesafe-sdk`

| | `ai[vercel]` (Vercel AI SDK for Python) | `typesafe-sdk` |
|---|---|---|
| Auth | One `AI_GATEWAY_API_KEY`, billing/logging/budgets via Vercel | Separate `TYPESAFE_API_KEY` + direct vendor account |
| Model access | `typesafe-ai/jev` gateway routing (fallbacks, observability, `providerMetadata.gateway`) | Direct `api.typesafe.ai` |
| API shape | `ai.ops.experimental_evaluate` with `ChoiceQuestion`/`ScoreQuestion`/`BooleanQuestion` | `Choice`/`Noul`/`Score` |
| Confidence | TypeSafe confidence surfaced at `providerMetadata["typesafe"]["confidence"]` | Native field |

Both packages are installed in this project; the engine uses the `ai` package
because the project is gateway-first and the gateway key was already the
machine's only credential. Swapping to `typesafe-sdk` later means changing only
`src/jev_classifier/engine.py`.

Note: the `ai` package wraps Jev's `noul` type as `BooleanQuestion` (`type:
"boolean"`); answers and probabilities match the native API.

## Setup

```bash
uv sync
cp .env.local.example .env.local   # then paste your AI Gateway key
```

Get the key from the [Vercel dashboard](https://vercel.com/dashboard) →
AI Gateway. The key is read from the `AI_GATEWAY_API_KEY` environment variable
(loaded automatically from `.env.local`).

## API capability test (verified 2026-09-20)

```bash
uv run python examples/smoke_test.py   # one state, all three question types
uv run python tests/test_api.py        # 10 labeled tickets, accuracy + invariants
```

Verified behavior of `typesafe-ai/jev` through the gateway:

- **Choice**: selected option + full probability distribution (sums to 1.0).
- **Score**: fractional probability-weighted score across ordered levels.
- **Boolean**: P(true) in [0, 1].
- **Confidence**: TypeSafe's separate confidence statistic at
  `providerMetadata["typesafe"]["confidence"]`, keyed by question id.
- **Usage**: input/output token counts returned per request.
- **Latency**: ~200 ms per evaluation (all questions in one parallel request).
- Accuracy on the 10-ticket sample: **8/10** (the two misses are tickets that
  legitimately straddle two labels).

## GUI

```bash
uv run python -m jev_classifier
```

Four tabs, mirroring ollama-classifier-gui:

1. **Settings** — model id (default `typesafe-ai/jev`), API-key status, toggle
   the extra urgency (Score) and complaint (Boolean) questions.
2. **Data** — load a CSV, pick the text column, preview rows.
3. **Schema** — one label per line as `name | description`.
4. **Results** — run classification row by row with live progress, inspect
   predicted label / confidence / urgency / complaint probability, export CSV.

On Linux the file dialogs require `zenity` (`sudo apt install zenity`).

## Project layout

```
src/jev_classifier/
  engine.py        # classify()/classify_async(): choice+score+boolean in one request
  gui.py           # Flet 1.0 desktop app
examples/smoke_test.py
tests/test_api.py  # labeled-dataset capability test
tests/data/test_tickets.csv
```
