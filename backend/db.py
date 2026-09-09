"""
Storage layer. Facts are stored as flexible JSON documents (not fixed columns)
so the schema can evolve as new kinds of facts appear across arbitrary PDFs.
"""
import sqlite3
import json
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "storage", "facts.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            filename TEXT,
            uploaded_at TEXT,
            num_pages INTEGER,
            status TEXT DEFAULT 'processing'
        )
    """)

    # Facts stored with a flexible JSON payload. Core fields that matter for
    # matching/comparison are pulled into columns for indexing; everything
    # else (arbitrary attributes the LLM decides matter) lives in `attributes`.
    c.execute("""
        CREATE TABLE IF NOT EXISTS facts (
            id TEXT PRIMARY KEY,
            document_id TEXT,
            page_number INTEGER,
            subject TEXT,          -- entity/topic the fact is about, e.g. "India GDP growth"
            metric TEXT,           -- normalized metric name, e.g. "real_gdp_growth_rate"
            value TEXT,            -- raw value as stated, e.g. "6.5%"
            numeric_value REAL,    -- parsed numeric value if applicable, for comparison
            unit TEXT,             -- %, USD bn, INR cr, etc.
            period TEXT,           -- time period/scope this fact applies to, e.g. "FY2024-25"
            evidence_text TEXT,    -- verbatim quoted span supporting the fact
            fact_type TEXT,        -- semantic | numerical
            attributes TEXT,       -- JSON blob of any extra LLM-identified attributes
            embedding TEXT,        -- JSON list of floats, for semantic matching
            created_at TEXT,
            FOREIGN KEY(document_id) REFERENCES documents(id)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS relationships (
            id TEXT PRIMARY KEY,
            fact_a_id TEXT,
            fact_b_id TEXT,
            relationship_type TEXT,  -- corroborates | contradicts | reconcilable | unrelated
            explanation TEXT,
            confidence REAL,
            created_at TEXT,
            FOREIGN KEY(fact_a_id) REFERENCES facts(id),
            FOREIGN KEY(fact_b_id) REFERENCES facts(id)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS extraction_issues (
            id TEXT PRIMARY KEY,
            document_id TEXT,
            page_number INTEGER,
            issue_type TEXT,     -- e.g. "ambiguous_unit", "ocr_garble", "low_confidence"
            description TEXT,
            raw_snippet TEXT,
            created_at TEXT
        )
    """)

    conn.commit()
    conn.close()


def now():
    return datetime.utcnow().isoformat()


def insert_document(doc_id, filename, num_pages):
    conn = get_conn()
    conn.execute(
        "INSERT INTO documents (id, filename, uploaded_at, num_pages, status) VALUES (?, ?, ?, ?, ?)",
        (doc_id, filename, now(), num_pages, "processing"),
    )
    conn.commit()
    conn.close()


def update_document_status(doc_id, status):
    conn = get_conn()
    conn.execute("UPDATE documents SET status=? WHERE id=?", (status, doc_id))
    conn.commit()
    conn.close()


def insert_fact(fact):
    conn = get_conn()
    conn.execute(
        """INSERT INTO facts
           (id, document_id, page_number, subject, metric, value, numeric_value,
            unit, period, evidence_text, fact_type, attributes, embedding, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            fact["id"], fact["document_id"], fact["page_number"], fact["subject"],
            fact["metric"], fact["value"], fact.get("numeric_value"), fact.get("unit"),
            fact.get("period"), fact["evidence_text"], fact["fact_type"],
            json.dumps(fact.get("attributes", {})),
            json.dumps(fact.get("embedding", [])),
            now(),
        ),
    )
    conn.commit()
    conn.close()


def get_all_facts(exclude_document_id=None):
    conn = get_conn()
    if exclude_document_id:
        rows = conn.execute(
            "SELECT * FROM facts WHERE document_id != ?", (exclude_document_id,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM facts").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_facts_by_document(document_id):
    conn = get_conn()
    rows = conn.execute("SELECT * FROM facts WHERE document_id=?", (document_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def insert_relationship(rel):
    conn = get_conn()
    conn.execute(
        """INSERT INTO relationships
           (id, fact_a_id, fact_b_id, relationship_type, explanation, confidence, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            rel["id"], rel["fact_a_id"], rel["fact_b_id"], rel["relationship_type"],
            rel["explanation"], rel.get("confidence", 0.5), now(),
        ),
    )
    conn.commit()
    conn.close()


def insert_issue(issue):
    conn = get_conn()
    conn.execute(
        """INSERT INTO extraction_issues
           (id, document_id, page_number, issue_type, description, raw_snippet, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            issue["id"], issue["document_id"], issue.get("page_number"),
            issue["issue_type"], issue["description"], issue.get("raw_snippet", ""),
            now(),
        ),
    )
    conn.commit()
    conn.close()


def get_all_documents():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM documents ORDER BY uploaded_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_relationships():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM relationships ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_issues():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM extraction_issues ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_fact(fact_id):
    conn = get_conn()
    row = conn.execute("SELECT * FROM facts WHERE id=?", (fact_id,)).fetchone()
    conn.close()
    return dict(row) if row else None
def clear_relationships():
    conn = get_conn()
    conn.execute("DELETE FROM relationships")
    conn.commit()
    conn.close()