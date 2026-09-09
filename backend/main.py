import os
import uuid
import shutil

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import db
import pdf_extract
import llm
import matching

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "storage", "uploads")
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Fact Knowledge Layer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

db.init_db()


@app.on_event("startup")
def startup():
    db.init_db()


@app.get("/")
def root():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


def _doc_name_map():
    docs = db.get_all_documents()
    return {d["id"]: d["filename"] for d in docs}


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    doc_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{doc_id}.pdf")
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        pages = pdf_extract.extract_pages(save_path)
    except Exception as e:
        raise HTTPException(400, f"Could not read PDF: {e}")

    db.insert_document(doc_id, file.filename, len(pages))

    chunks = pdf_extract.chunk_pages(pages, pages_per_chunk=3)

    new_facts = []
    extraction_errors = 0

    for i, chunk in enumerate(chunks):
        print(f"[upload] extracting chunk {i+1}/{len(chunks)} (pages {chunk['page_range']})", flush=True)
        raw_facts, error = llm.extract_facts_from_chunk(chunk["text"])
        if error:
            extraction_errors += 1
            db.insert_issue({
                "id": str(uuid.uuid4()),
                "document_id": doc_id,
                "page_number": chunk["page_range"][0],
                "issue_type": "extraction_failure",
                "description": error,
                "raw_snippet": chunk["text"][:200],
            })
            continue

        for rf in raw_facts:
            # Defensive validation: skip malformed fact objects rather than
            # crashing the whole upload. This is exactly the kind of
            # extraction failure the assignment asks us to surface.
            required = ["subject", "metric", "value", "evidence_text", "fact_type"]
            if not all(k in rf and rf[k] for k in required):
                db.insert_issue({
                    "id": str(uuid.uuid4()),
                    "document_id": doc_id,
                    "page_number": chunk["page_range"][0],
                    "issue_type": "malformed_fact",
                    "description": f"Fact object missing required fields: {rf}",
                    "raw_snippet": str(rf)[:200],
                })
                continue

                        # Verify evidence_text actually appears in the source chunk.
            # Whitespace is normalized, and a quote split by "..." is
            # checked as two separate substrings so an abbreviated-but-real
            # quote isn't wrongly flagged as fabricated.
            def _normalize(s):
                return " ".join(s.split())

            norm_chunk = _normalize(chunk["text"])
            raw_evidence = rf["evidence_text"].strip()
            evidence_parts = [p.strip() for p in raw_evidence.split("...") if p.strip()]
            if not evidence_parts:
                evidence_parts = [raw_evidence]

            is_grounded = all(_normalize(part) in norm_chunk for part in evidence_parts)

            if not is_grounded:
                db.insert_issue({
                    "id": str(uuid.uuid4()),
                    "document_id": doc_id,
                    "page_number": rf.get("page_number", chunk["page_range"][0]),
                    "issue_type": "ungrounded_evidence",
                    "description": (
                        "Model-provided evidence quote was not found verbatim "
                        "in the source text — possible hallucination."
                    ),
                    "raw_snippet": rf["evidence_text"][:200],
                })
                continue

            fact_id = str(uuid.uuid4())
            embedding = llm.embed_text(f"{rf['subject']} {rf['metric']} {rf.get('period','')}")

            fact = {
                "id": fact_id,
                "document_id": doc_id,
                "page_number": rf.get("page_number", chunk["page_range"][0]),
                "subject": rf["subject"],
                "metric": rf["metric"],
                "value": rf["value"],
                "numeric_value": rf.get("numeric_value"),
                "unit": rf.get("unit"),
                "period": rf.get("period"),
                "evidence_text": rf["evidence_text"],
                "fact_type": rf["fact_type"],
                "attributes": rf.get("attributes", {}),
                "embedding": embedding,
            }
            db.insert_fact(fact)
            new_facts.append(fact)

    db.update_document_status(doc_id, "extracted")

    # Cross-document comparison
    doc_names = _doc_name_map()
    print(f"[upload] extraction done: {len(new_facts)} facts, {extraction_errors} chunk errors. Running cross-document comparison...", flush=True)
    relationships = matching.process_new_document_facts(new_facts, doc_names)

    db.update_document_status(doc_id, "complete")

    return {
        "document_id": doc_id,
        "filename": file.filename,
        "num_pages": len(pages),
        "facts_extracted": len(new_facts),
        "chunks_processed": len(chunks),
        "extraction_errors": extraction_errors,
        "relationships_found": len(relationships),
    }


@app.get("/api/documents")
def list_documents():
    return db.get_all_documents()


@app.get("/api/facts")
def list_facts():
    facts = db.get_all_facts()
    doc_names = _doc_name_map()
    for f in facts:
        f["document_name"] = doc_names.get(f["document_id"], f["document_id"])
        f.pop("embedding", None)
    return facts


@app.get("/api/relationships")
def list_relationships():
    rels = db.get_all_relationships()
    doc_names = _doc_name_map()
    out = []
    for r in rels:
        fa = db.get_fact(r["fact_a_id"])
        fb = db.get_fact(r["fact_b_id"])
        if not fa or not fb:
            continue
        for f in (fa, fb):
            f["document_name"] = doc_names.get(f["document_id"], f["document_id"])
            f.pop("embedding", None)
        out.append({
            "id": r["id"],
            "relationship_type": r["relationship_type"],
            "explanation": r["explanation"],
            "confidence": r["confidence"],
            "fact_a": fa,
            "fact_b": fb,
        })
    return out


@app.get("/api/issues")
def list_issues():
    return db.get_all_issues()


@app.post("/api/recompute-relationships")
def recompute_relationships():
    doc_names = _doc_name_map()
    rels = matching.recompute_all_relationships(doc_names)
    return {"relationships_found": len(rels)}


@app.get("/api/summary")
def summary():
    facts = db.get_all_facts()
    rels = db.get_all_relationships()
    issues = db.get_all_issues()
    counts = {"corroborates": 0, "contradicts": 0, "reconcilable": 0}
    for r in rels:
        if r["relationship_type"] in counts:
            counts[r["relationship_type"]] += 1
    return {
        "total_documents": len(db.get_all_documents()),
        "total_facts": len(facts),
        "total_relationships": len(rels),
        "total_issues": len(issues),
        "relationship_counts": counts,
    }
