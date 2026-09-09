"""
Finds candidate matching facts across documents and asks the LLM to judge
their relationship. Matching is deliberately cheap/approximate (embedding
cosine similarity + metric name overlap) since it's just a shortlist step —
the actual corroborate/contradict/reconcile judgment is made by the LLM,
which is where the real reasoning needs to happen.
"""
import json
import uuid
import re
import numpy as np

import db
import llm

SIMILARITY_THRESHOLD = 0.45


def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _as_list(embedding):
    """Embeddings arrive as a JSON string when read back from the DB, but
    as a plain Python list for freshly-built facts not yet persisted.
    Accept either."""
    if isinstance(embedding, str):
        return json.loads(embedding)
    return embedding


def _keyword_overlap(a_fact, b_fact):
    """Cheap lexical signal: how much do the subject+metric words overlap?
    Complements the hashed embedding, which is noisy on its own."""
    def words(f):
        text = f"{f.get('subject','')} {f.get('metric','')}".lower()
        return set(re.findall(r"[a-z]+", text))
    wa, wb = words(a_fact), words(b_fact)
    if not wa or not wb:
        return 0.0
    overlap = len(wa & wb) / max(1, min(len(wa), len(wb)))
    return overlap


def find_candidate_matches(new_fact, existing_facts):
    """Returns a list of existing facts likely to be about the same
    underlying real-world fact as new_fact, ranked by similarity."""
    new_emb = _as_list(new_fact["embedding"])
    candidates = []
    for ef in existing_facts:
        if ef["id"] == new_fact["id"]:
            continue  # never compare a fact to itself
        ef_emb = _as_list(ef["embedding"])
        sim = cosine_sim(new_emb, ef_emb)

        # Boost similarity if metric names match exactly or subject overlaps —
        # cheap signal that's often more reliable than the hashed embedding alone.
        if ef["metric"] and new_fact["metric"] and ef["metric"] == new_fact["metric"]:
            sim += 0.25

        sim += 0.3 * _keyword_overlap(new_fact, ef)

        if sim >= SIMILARITY_THRESHOLD:
            candidates.append((sim, ef))

    candidates.sort(key=lambda x: -x[0])
    return [c[1] for c in candidates[:4]]  # cap to top 4 to limit LLM calls


def process_new_document_facts(new_facts, doc_names):
    """
    For every newly extracted fact, find candidate matches among all
    previously stored facts (across all documents), and run an LLM
    comparison on each candidate pair. Stores resulting relationships.
    """
    all_existing = db.get_all_facts()
    relationships_created = []
    unrelated_count = 0

    for new_fact in new_facts:
        candidates = find_candidate_matches(new_fact, all_existing)
        for cand in candidates:
            result, error = llm.compare_facts(new_fact, cand, doc_names)
            if error:
                db.insert_issue({
                    "id": str(uuid.uuid4()),
                    "document_id": new_fact["document_id"],
                    "page_number": new_fact["page_number"],
                    "issue_type": "comparison_failure",
                    "description": error,
                    "raw_snippet": f"{new_fact['id']} vs {cand['id']}",
                })
                continue

            if result["relationship_type"] == "unrelated":
                # Skipped, but logged (not silently) so a spike in "unrelated"
                # verdicts is visible instead of just showing up as a drop
                # in relationship count with no explanation anywhere.
                unrelated_count += 1
                print(f"[matching] unrelated: {new_fact['id']} vs {cand['id']} — {result.get('explanation','')[:200]}")
                continue

            rel = {
                "id": str(uuid.uuid4()),
                "fact_a_id": new_fact["id"],
                "fact_b_id": cand["id"],
                "relationship_type": result["relationship_type"],
                "explanation": result["explanation"],
                "confidence": result.get("confidence", 0.5),
            }
            db.insert_relationship(rel)
            relationships_created.append(rel)

        # Add the new fact to the pool so later facts in this same batch
        # can also be compared against it.
        all_existing.append(new_fact)

    if unrelated_count:
        print(f"[matching] {unrelated_count} candidate pair(s) judged unrelated and skipped this run.")

    return relationships_created


def recompute_all_relationships(doc_names):
    """
    Wipes and rebuilds all relationships from scratch across every fact
    currently stored, using the current matching thresholds/logic. Useful
    when matching parameters change and you don't want to re-upload (and
    re-pay the extraction cost for) every PDF.
    """
    db.clear_relationships()
    all_facts = db.get_all_facts()
    relationships_created = []
    seen_pairs = set()

    for i, fact in enumerate(all_facts):
        pool = all_facts[:i] + all_facts[i+1:]
        candidates = find_candidate_matches(fact, pool)
        for cand in candidates:
            pair_key = tuple(sorted([fact["id"], cand["id"]]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            result, error = llm.compare_facts(fact, cand, doc_names)
            if error:
                db.insert_issue({
                    "id": str(uuid.uuid4()),
                    "document_id": fact["document_id"],
                    "page_number": fact["page_number"],
                    "issue_type": "comparison_failure",
                    "description": error,
                    "raw_snippet": f"{fact['id']} vs {cand['id']}",
                })
                continue

            if result["relationship_type"] == "unrelated":
                print(f"[matching] unrelated: {fact['id']} vs {cand['id']} — {result.get('explanation','')[:200]}")
                continue

            rel = {
                "id": str(uuid.uuid4()),
                "fact_a_id": fact["id"],
                "fact_b_id": cand["id"],
                "relationship_type": result["relationship_type"],
                "explanation": result["explanation"],
                "confidence": result.get("confidence", 0.5),
            }
            db.insert_relationship(rel)
            relationships_created.append(rel)

    return relationships_created