"""
LLM interface (Groq). Two jobs:
  1. Extract structured facts from a chunk of PDF text.
  2. Judge the relationship between two candidate-matching facts
     (corroborates / contradicts / reconcilable / unrelated).

We ask for strict JSON back and parse defensively, since a flaky/partial
JSON response from the model is one of the realistic failure modes for
this kind of pipeline (see the "extraction failure" requirement).
"""
import os
import json
import re
import time
from groq import Groq

MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")

_client = None


def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable is not set")
        _client = Groq(api_key=api_key)
    return _client


def _extract_json(text):
    """Best-effort extraction of a JSON object/array from an LLM response
    that may include stray prose, markdown fences, etc."""
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text.strip())
    text = re.sub(r"```$", "", text.strip())
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fallback: find the first {...} or [...] block
    for pattern in (r"\[.*\]", r"\{.*\}"):
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                continue
    return None


EXTRACTION_PROMPT = """You are a fact extraction engine. You are given a chunk of text from a PDF document (page numbers are marked inline as [PAGE N]).

Extract every meaningful, checkable FACT from this text. A fact is a specific, verifiable claim — usually numerical (a figure, rate, amount, date, count) or a specific semantic claim (a status, relationship, or event). Do NOT extract vague statements, opinions, or boilerplate.

For each fact, output an object with these fields:
- "subject": the entity/topic the fact is about (e.g. "India real GDP growth", "Delhivery revenue from operations")
- "metric": a normalized, machine-friendly metric name using snake_case (e.g. "real_gdp_growth_rate", "revenue_from_operations"). Use consistent naming so the same underlying metric gets the same name even if phrased differently.
- "value": the value as stated in the text (e.g. "6.5%", "INR 8,141 crore")
- "numeric_value": the value as a plain float if it is numeric (no units/commas), else null
- "unit": the unit if any (e.g. "%", "INR crore", "USD billion"), else null
- "period": the time period or scope this fact applies to, as stated (e.g. "FY2024-25", "Q4 FY24", "as of March 2025"), else null if not specified
- "page_number": the page number (integer) this fact was found on, from the nearest [PAGE N] marker
- "evidence_text": a SHORT VERBATIM quote (max ~30 words) from the source text that directly supports this fact. Must be an exact substring of the given text.
- "fact_type": either "numerical" or "semantic"
- "attributes": an object with any other relevant context you think matters (e.g. {{"source_type": "government report"}}, {{"basis": "constant prices"}}). Keep this small and only include what's genuinely useful.

Only extract facts that are clearly, explicitly stated. Do not infer or calculate values not directly present. Do not extract table-of-contents entries, headers, or page furniture as facts.

Return ONLY a JSON array of fact objects. If there are no extractable facts in this text, return an empty array [].

TEXT:
{text}
"""
def extract_facts_from_chunk(text, retries=2):
    client = get_client()
    # Chunks are now pre-capped to ~4500 chars by pdf_extract.chunk_pages, but
    # this slice remains as a hard safety net so a single call's prompt
    # tokens + max_tokens completion budget can't exceed a small provider's
    # per-minute token limit (e.g. Groq free tier's 8000 TPM) on its own.
    char_limit = 6000
    last_error = None
    for attempt in range(retries + 1):
        prompt = EXTRACTION_PROMPT.format(text=text[:char_limit])
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2500,
            )
            raw = resp.choices[0].message.content
            parsed = _extract_json(raw)
            if parsed is None:
                last_error = f"Could not parse JSON from model response: {raw[:300]}"
                continue
            if not isinstance(parsed, list):
                last_error = f"Expected a JSON array, got: {type(parsed)}"
                continue
            return parsed, None
        except Exception as e:
            err = str(e)
            last_error = f"LLM extraction call failed: {e}"
            if "tokens per minute" in err.lower() or "Error code: 413" in err:
                # This single request was too big for the per-minute cap
                # regardless of daily quota — shrink it and retry
                # immediately rather than waiting out a cooldown that isn't
                # actually the problem here.
                char_limit = max(1500, char_limit // 2)
                continue
            if "429" in err or "rate_limit" in err.lower():
                time.sleep(8)
            else:
                time.sleep(1.5)
        finally:
            time.sleep(2.5)
    return [], last_error


COMPARISON_PROMPT = """You are comparing two facts extracted from different (or the same) documents to determine their relationship.

FACT A (from document "{doc_a}", page {page_a}):
Subject: {subject_a}
Metric: {metric_a}
Value: {value_a} {unit_a}
Period/Scope: {period_a}
Evidence: "{evidence_a}"

FACT B (from document "{doc_b}", page {page_b}):
Subject: {subject_b}
Metric: {metric_b}
Value: {value_b} {unit_b}
Period/Scope: {period_b}
Evidence: "{evidence_b}"

Hint (computed, not authoritative): {numeric_hint}

Work through these questions IN ORDER and stop at the first one that applies. Do not skip ahead to "reconcilable" just because you can imagine some hypothetical context — only use it if the period, scope, or unit difference is ACTUALLY STATED in the two facts above, not merely conceivable.

1. Do the two facts concern a genuinely different subject or metric (not just a different period/value of the SAME subject+metric)? For example, "GDP growth" vs "inflation rate" is unrelated; "GDP growth for 2023" vs "GDP growth for 2024" is NOT unrelated (that's case 4 below). -> only then, "unrelated"
2. Do Period/Scope and unit/basis both match (or both are unspecified/equivalent), AND the values are the same or trivially different (rounding, formatting, different phrasing of an identical number/status)? -> "corroborates"
3. Do Period/Scope and unit/basis both match (or both are unspecified/equivalent), AND the values are meaningfully different (not just rounding) or the statuses conflict, with nothing in the text above explaining the gap? -> "contradicts"
4. Only if the Period/Scope, unit, or basis are EXPLICITLY different between A and B (e.g. one says "FY23" and the other says "FY24"; one says "consolidated" and the other "standalone"; different currencies or nominal-vs-real) and that stated difference plausibly accounts for the differing values -> "reconcilable"

"unrelated" should be rare — we only send you pairs that already passed a similarity shortlist, so default to treating same-subject-different-period/scope facts as case 4 ("reconcilable"), NOT case 1 ("unrelated"). Only use "unrelated" when the subject or metric itself is genuinely different, not when it's the same subject/metric measured at a different time or scope.

"reconcilable" requires a concrete, stated difference you can point to (quote the specific words from Period/Scope or evidence that differ) — never choose it purely because you can't rule out an unstated explanation. When in doubt between "contradicts" and "reconcilable", check: is the reconciling factor actually written in the two facts above? If not, it's "contradicts".

Examples:
- A: "revenue FY2024 = INR 8,141 crore" vs B: "revenue for FY2024 = INR 8,141.2 crore" (same period, same basis, rounding only) -> corroborates
- A: "net profit FY2024 = INR 100 crore" vs B: "net profit FY2024 = INR 250 crore" (same period, same scope, no stated reason for gap) -> contradicts
- A: "revenue FY2023 = INR 5,000 crore" vs B: "revenue FY2024 = INR 8,141 crore" (different periods explicitly stated) -> reconcilable
- A: "consolidated revenue = INR 8,141 crore" vs B: "standalone revenue = INR 6,900 crore" (different stated scope) -> reconcilable

Respond ONLY with a JSON object:
{{
  "relationship_type": "corroborates" | "contradicts" | "reconcilable" | "unrelated",
  "explanation": "1-3 sentences explaining your reasoning, referencing the specific values/periods/scopes involved, and for reconcilable/contradicts explicitly naming which fields (period/scope/unit) match or differ",
  "confidence": 0.0 to 1.0
}}
"""
def _numeric_hint(fact_a, fact_b):
    """Deterministic pre-check to steer the LLM away from defaulting to
    'reconcilable' — computed in code so it's not subject to the model's
    own hedging."""
    period_a = (fact_a.get("period") or "").strip().lower()
    period_b = (fact_b.get("period") or "").strip().lower()
    unit_a = (fact_a.get("unit") or "").strip().lower()
    unit_b = (fact_b.get("unit") or "").strip().lower()
    same_period = period_a == period_b
    same_unit = unit_a == unit_b

    na, nb = fact_a.get("numeric_value"), fact_b.get("numeric_value")
    if na is not None and nb is not None:
        try:
            na, nb = float(na), float(nb)
        except (TypeError, ValueError):
            na = nb = None
        if na is not None and nb is not None:
            denom = max(abs(na), abs(nb), 1e-9)
            pct_diff = abs(na - nb) / denom
            if same_period and same_unit:
                if pct_diff < 0.02:
                    return "period and unit both match and numeric values are within ~2% of each other — likely corroborates, not reconcilable."
                else:
                    return (f"period and unit both match but numeric values differ by ~{pct_diff*100:.0f}% "
                            "with no stated period/scope/unit difference — likely contradicts, not reconcilable.")
            else:
                return "period and/or unit differ between the two facts as stated — a reconcilable classification should name that exact difference."
    if not same_period or not same_unit:
        return "period and/or unit differ between the two facts as stated — a reconcilable classification should name that exact difference."
    return "period and unit both match or are both unspecified — check whether values genuinely agree (corroborates) or conflict (contradicts)."


def compare_facts(fact_a, fact_b, doc_names):
    client = get_client()
    prompt = COMPARISON_PROMPT.format(
        doc_a=doc_names.get(fact_a["document_id"], fact_a["document_id"]),
        page_a=fact_a["page_number"],
        subject_a=fact_a["subject"], metric_a=fact_a["metric"],
        value_a=fact_a["value"], unit_a=fact_a.get("unit") or "",
        period_a=fact_a.get("period") or "unspecified",
        evidence_a=fact_a["evidence_text"],
        doc_b=doc_names.get(fact_b["document_id"], fact_b["document_id"]),
        page_b=fact_b["page_number"],
        subject_b=fact_b["subject"], metric_b=fact_b["metric"],
        value_b=fact_b["value"], unit_b=fact_b.get("unit") or "",
        period_b=fact_b.get("period") or "unspecified",
        evidence_b=fact_b["evidence_text"],
        numeric_hint=_numeric_hint(fact_a, fact_b),
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1024,
        )
        raw = resp.choices[0].message.content
        parsed = _extract_json(raw)
        if parsed is None or "relationship_type" not in parsed:
            # Response likely got cut off mid-JSON before hitting the
            # closing brace. Retry once with a harder instruction to be
            # terse, rather than silently losing the pair.
            retry_prompt = prompt + "\n\nIMPORTANT: Keep your explanation to ONE short sentence so the full JSON object fits well within the token limit."
            try:
                resp2 = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": retry_prompt}],
                    temperature=0.1,
                    max_tokens=1024,
                )
                raw2 = resp2.choices[0].message.content
                parsed = _extract_json(raw2)
                if parsed is None or "relationship_type" not in parsed:
                    return None, f"Could not parse comparison response after retry: {raw2[:300]}"
                return parsed, None
            except Exception:
                return None, f"Could not parse comparison response: {raw[:300]}"
        return parsed, None
    except Exception as e:
        return None, f"LLM comparison call failed: {e}"
    finally:
        time.sleep(1.2)


def embed_text(text):
    """
    Groq doesn't serve embedding models, so we use a lightweight local
    TF-IDF-style hashing embedding for candidate-matching purposes only.
    This is a deliberate trade-off: it's good enough to shortlist similar
    facts by subject/metric wording, and the real judgment (corroborates /
    contradicts / reconcilable) is left to the LLM comparison step, which
    is far more reliant on reasoning than embedding precision.
    """
    import hashlib
    import numpy as np

    dim = 256
    vec = np.zeros(dim)
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    for tok in tokens:
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        idx = h % dim
        sign = 1 if (h // dim) % 2 == 0 else -1
        vec[idx] += sign
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()