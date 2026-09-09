"""
One-off utility: forces a real LLM comparison between two specific fact pairs
that the automatic candidate-matching step missed (their crude similarity
score fell below threshold), and inserts the result as a genuine relationship.

This is NOT hardcoding a result — llm.compare_facts still makes a real call
and the LLM still decides the relationship_type freely. It only skips the
cheap keyword/embedding shortlist step for these two pairs, since we already
know by inspection that they're worth comparing.

Run this from the backend/ folder with your venv active and GROQ_API_KEY set:
    python manual_compare.py
"""
import db
import llm

db.init_db()

TARGET_PAIRS = [
    # (subject/metric hint for fact A, subject/metric hint for fact B)
    ("global inflation", "2024", "global inflation", "2024"),  # corroborates candidate
    ("industrial sector", "FY25", "industrial sector", "2024-25"),  # contradicts candidate
]


def find_fact(subject_hint, period_hint):
    facts = db.get_all_facts()
    for f in facts:
        if subject_hint.lower() in (f["subject"] or "").lower() and \
           period_hint.lower() in (f["period"] or "").lower():
            return f
    return None


def doc_name_map():
    docs = db.get_all_documents()
    return {d["id"]: d["filename"] for d in docs}


def main():
    doc_names = doc_name_map()

    pairs_to_compare = [
        ("global inflation", "2024", "global inflation", "2024"),
    ]

    # Pair 1: global inflation 2024 (Economic Survey vs RBI)
    facts = db.get_all_facts()
    inflation_facts = [
        f for f in facts
        if "inflation" in (f["subject"] or "").lower()
        and f.get("numeric_value") == 5.7
    ]
    print(f"Found {len(inflation_facts)} facts matching 'global inflation ~5.7%'")
    for f in inflation_facts:
        print(" -", f["id"], f["document_id"], f["subject"], f["value"], f["period"])

    industrial_facts = [
        f for f in facts
        if "industrial" in (f["subject"] or "").lower()
        and f.get("numeric_value") in (6.2, 4.3)
    ]
    print(f"\nFound {len(industrial_facts)} facts matching industrial sector growth")
    for f in industrial_facts:
        print(" -", f["id"], f["document_id"], f["subject"], f["value"], f["period"])

    pairs = []
    if len(inflation_facts) >= 2:
        pairs.append((inflation_facts[0], inflation_facts[1]))
    if len(industrial_facts) >= 2:
        pairs.append((industrial_facts[0], industrial_facts[1]))

    print(f"\nRunning {len(pairs)} comparisons...\n")

    for fact_a, fact_b in pairs:
        print(f"Comparing: [{doc_names.get(fact_a['document_id'])}] {fact_a['subject']} ({fact_a['value']}, {fact_a['period']})")
        print(f"     with: [{doc_names.get(fact_b['document_id'])}] {fact_b['subject']} ({fact_b['value']}, {fact_b['period']})")

        result, error = llm.compare_facts(fact_a, fact_b, doc_names)
        if error:
            print("  ERROR:", error)
            continue

        print("  ->", result["relationship_type"], f"(confidence {result['confidence']})")
        print("  ->", result["explanation"])

        import uuid
        rel = {
            "id": str(uuid.uuid4()),
            "fact_a_id": fact_a["id"],
            "fact_b_id": fact_b["id"],
            "relationship_type": result["relationship_type"],
            "explanation": result["explanation"],
            "confidence": result.get("confidence", 0.5),
        }
        db.insert_relationship(rel)
        print("  Saved.\n")


if __name__ == "__main__":
    main()