# Ledger — a Fact Knowledge Layer

Extracts facts from PDFs, grounds every fact in its source evidence, and finds
where facts across documents corroborate, contradict, or can be reconciled
through context (different time periods, scopes, or units).

## Setup and Run

Requirements: Python 3.10+, a free [Groq](https://console.groq.com) API key.

```bash
git clone <your-repo-url>
cd fact-knowledge-layer

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt

cp backend/.env.example backend/.env
# edit backend/.env and paste your real GROQ_API_KEY

cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000` in a browser. Drag PDFs into the upload panel.
Upload at least two documents covering overlapping topics (the three starter
datasets in `india-macroeconomy/` all work well together) to see
cross-document relationships appear.

The whole app is a single FastAPI server + a static HTML/JS frontend it
serves directly — no separate frontend build step, no external database
(SQLite file at `storage/facts.db`, created automatically).

**Note on the LLM model:** this project uses Groq's `openai/gpt-oss-120b`
model (configurable via the `GROQ_MODEL` environment variable). During
development, Groq deprecated the originally-used `llama-3.3-70b-versatile`
model mid-build, which is exactly the kind of external dependency risk worth
flagging — see Limitations below.

## Video Demo

[Watch the demo video](https://drive.google.com/file/d/1fifZWgEcfNW4rWAVvS_fZBYpEHuqRWfC/view?usp=sharing)

## Approach

**Extraction.** Each PDF is parsed page-by-page (PyMuPDF), then grouped into
3-page chunks and sent to an LLM with a prompt asking it to pull out specific,
checkable facts — not vague statements — each with a normalized metric name,
value, unit, period/scope, and a verbatim quoted evidence span. Chunking
(rather than single pages or the whole document) is the trade-off point:
enough context to catch facts split across a page break, without blowing
past context limits or losing evidence precision.

**Grounding.** Every extracted fact's `evidence_text` is checked against the
literal source chunk before it's accepted. Whitespace is normalized on both
sides (PDF extraction and model output don't always agree on line breaks),
and a quote split by an ellipsis ("...") is checked as two separate
substrings, so an abbreviated-but-real quote isn't wrongly flagged. If a
quote still doesn't match, the fact is rejected and logged as an
`ungrounded_evidence` issue instead of silently trusted. This is a direct
defense against LLM hallucination.

**Matching facts across documents.** Facts aren't compared pairwise against
everything (that's O(n²) LLM calls). Each fact is embedded with a lightweight
local hashing embedding over subject/metric/period (Groq doesn't serve a
hosted embedding model), compared by cosine similarity against previously
stored facts, and boosted by exact metric-name matches and lexical keyword
overlap. Only the top candidates per fact go to the LLM for a real judgment
call, keeping cost roughly linear in the number of facts rather than
quadratic.

**Relationship judgment.** For each candidate pair, the LLM chooses exactly
one of: corroborates, contradicts, reconcilable (context explains an apparent
difference — different period, scope, or unit), or unrelated (a false
positive from the shortlist step, discarded). It gives a short explanation
citing the specific values/periods/scopes involved.

**Dynamic schema.** Facts are stored with a small set of indexed columns
(subject, metric, value, period, etc.) plus an open `attributes` JSON blob
for anything else the model decides is relevant. Nothing about the schema
assumes a particular document type or domain.

**AI tools used.** Claude (Anthropic) for the full build — schema design,
extraction/comparison prompts, the matching pipeline, debugging, and the
frontend. Llama-family models via Groq (specifically `openai/gpt-oss-120b`,
after `llama-3.3-70b-versatile` was deprecated mid-build) are the runtime LLM
used by the deployed app itself for extraction and fact comparison.

## The Four Required Cases

Demonstrated on the `india-macroeconomy` starter dataset (Economic Survey
2024-25, RBI Annual Report 2024-25, IMF Article IV 2025) — three independent
sources reporting overlapping macroeconomic indicators.

**1. Corroborated fact** — Global inflation for 2024, stated as **5.7%** in
both the Economic Survey ("to 5.7 per cent in 2024", p.76) and the RBI Annual
Report ("Global inflation eased to 5.7 per cent in 2024", p.7) — same figure,
same year, worded independently by two different sources.

**2. Genuine contradiction** — India's industrial sector growth for the same
fiscal year: the Economic Survey's First Advance Estimate states **6.2%**
("The industrial sector is estimated to grow by 6.2 per cent in FY25", p.14),
while the RBI Annual Report states **4.3%** ("Growth in industrial sector
moderated to 4.3 per cent in 2024-25", p.9). Same metric, same fiscal year, a
~2 percentage point gap, with neither document explaining the discrepancy —
flagged rather than silently averaged away.

**3. Context-explained apparent contradiction** — Global real GDP growth
projections differ across the same pair of documents purely because of
differing time horizons: the Economic Survey cites an "around 3.2 per cent"
average over the *next five years* (p.5), while the RBI report cites 3.3%
for *2024 alone* and 3.7% as the *2000-2019 historical average* (p.6). The
system correctly classifies these as reconcilable rather than contradictory
once it identifies the different time scopes involved.

**4. Extraction/reasoning failure** — Logged and visible in the "Extraction
Issues" tab of the UI:
   - A real `429` rate-limit failure from Groq's free tier during a large
     (89-page) document upload.
   - Several cases where the model's JSON response was truncated mid-array,
     losing facts from that chunk.
   - Multiple cases where the grounding check (see Approach) rejected
     genuinely true facts because the model paraphrased its quote instead of
     reproducing it exactly — a real false-negative in the grounding logic,
     not a hallucination. One concrete instance: the Economic Survey's own
     statement that "the real gross domestic product (GDP) growth for FY25
     is estimated to be 6.4 per cent" was extracted but rejected by the
     grounding check due to a minor phrasing mismatch, and never made it
     into the stored fact set — a false negative worth fixing next.

## Limitations and Next Steps

- **Embedding quality.** Candidate-matching uses a cheap local hashing
  embedding rather than a real semantic embedding model (Groq doesn't serve
  one). This caused real problems during testing — one broad, generically-
  worded fact ("global economy... average growth... next five years")
  dominated the match pool and crowded out more specific true matches
  between the documents. A real sentence-embedding model would be the first
  upgrade.
- **Grounding check false negatives.** As documented in case 4 above, the
  strict evidence-matching check has rejected true facts due to minor
  paraphrasing. The current fix (whitespace normalization, ellipsis-aware
  splitting) helped but didn't eliminate this; a fuzzy-match threshold
  (e.g. edit distance) would catch more real facts without reopening the
  door to hallucinated ones.
- **Free-tier rate limits.** Sequential chunk processing with pacing delays
  avoids crashing on Groq's free tier, but makes large-document processing
  slow (minutes per document). A paid tier or parallelized requests with
  proper backoff would fix this for production use.
- **Table extraction.** PyMuPDF's plain text extraction linearizes tables,
  sometimes scrambling numeric alignment in dense financial tables. A
  layout-aware extractor would improve accuracy on tabular data.
- **No incremental re-ranking.** If a third document changes the correct
  interpretation of an existing "contradicts" relationship into
  "reconcilable," the system doesn't automatically revisit it — relationships
  are only computed forward from new facts, or via a manual full recompute.
- **Confidence calibration.** The LLM's self-reported confidence score isn't
  independently validated — useful for sorting, not a calibrated probability.

## Additional Notes

Built iteratively with real debugging along the way: an initial LLM model
choice (`llama-3.3-70b-versatile`) was deprecated by Groq mid-build and had
to be swapped for `openai/gpt-oss-120b`; a real rate-limit crash on a large
PDF led to adding retry/backoff and pacing; and the matching threshold was
tuned after an early run surfaced zero relationships due to overly strict
similarity scoring. These are documented as real engineering decisions
above, not smoothed over.
