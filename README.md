# Financial Knowledge Base

A versioned ingestion and extraction pipeline for SEC filings.

The current pipeline discovers a filing, downloads its source HTML, archives the
raw bytes with metadata and a SHA-256 hash, and produces deterministic narrative
sections for later retrieval and AI extraction.

## Setup

```bash
uv sync --dev
```

SEC requests require an identifying user agent with a contact address:

```bash
export SEC_USER_AGENT="FinancialKnowledgeBase/0.1 you@example.com"
```

## Ingest a filing

```bash
uv run financial-knowledge-base ingest \
  --cik 0000320193 \
  --form 10-Q
```

Each filing is stored under `data/raw/{cik}/{accession-number}/`:

- `filing.html`: immutable source document
- `metadata.json`: SEC metadata, download time, size, and source hash
- `document.json`: normalized text, parser version, text hash, and Item sections

## Reparse an archived filing

Parsing is independent from downloading. This lets a new parser version rebuild
derived documents without changing the original evidence.

```bash
uv run financial-knowledge-base parse \
  --filing-path data/raw/0000320193/0000320193-26-000020/filing.html
```

## Extract Item 2 claims

The extraction command sends only Part I, Item 2 to a model and requests a
validated structured response. Every returned evidence quote is checked against
the parsed document before the proposal is stored.

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_MODEL="a-model-that-supports-structured-outputs"

uv run financial-knowledge-base extract-item-2 \
  --document-path data/raw/0000320193/0000320193-26-000020/document.json
```

Proposals are immutable and stored under
`extractions/item-2/{prompt-version}/{response-id}.json` inside the filing
directory. A proposal does not update the knowledge base automatically.

## Review and evaluate an extraction

Create a review template from a proposal:

```bash
uv run financial-knowledge-base create-item-2-review \
  --proposal-path path/to/proposal.json \
  --reviewer your-name
```

Edit the generated review JSON. Set each claim's `decision` to `accepted`,
`rejected`, or `needs_edit`. Rejected claims require a `rejection_reason`, and
edited claims require a complete `corrected_claim`. Add claims the model omitted
to `missed_claims`.

Finalize the review as a reusable, checked-in gold case:

```bash
uv run financial-knowledge-base finalize-item-2-review \
  --proposal-path path/to/proposal.json \
  --review-path path/to/proposal.review.json \
  --document-path path/to/document.json \
  --case-name apple-2026-q3
```

Score any proposal for the same source document:

```bash
uv run financial-knowledge-base evaluate-item-2 \
  --proposal-path path/to/proposal.json \
  --gold-case-path evals/item-2/apple-2026-q3.json
```

The evaluator matches claims by exact evidence quote and reports precision,
recall, F1, claim-type accuracy, exact-field match rate, evidence-verification
rate, ungrounded-claim rate, and unmatched-claim rate. Human review remains
responsible for judging whether a claim is material and semantically correct.
