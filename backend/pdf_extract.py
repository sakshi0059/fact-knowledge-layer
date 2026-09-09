"""
Extracts per-page text from a PDF. Kept intentionally simple: page-level text
blocks are enough for LLM-based fact extraction and give us clean, precise
evidence pointers (document + page number) without needing layout analysis.
"""
import pymupdf as fitz  # PyMuPDF


def extract_pages(pdf_path):
    """Returns a list of dicts: [{page_number, text}, ...], 1-indexed pages."""
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        pages.append({"page_number": i + 1, "text": text})
    doc.close()
    return pages


def chunk_pages(pages, pages_per_chunk=3, max_chars_per_chunk=4500):
    """
    Groups consecutive pages into chunks for LLM extraction. Chunking (rather
    than one page at a time) lets facts that span a sentence broken across a
    page boundary get captured, and is more token-efficient. Each chunk keeps
    track of which page numbers it covers so evidence can still be attributed
    to a specific page.

    max_chars_per_chunk caps chunk size so that a single extraction call
    (prompt + this text + requested completion tokens) stays comfortably
    under a small provider's tokens-per-minute limit (e.g. Groq free tier's
    8000 TPM). Dense report pages (financial tables, IMF/RBI-style text) can
    individually exceed this cap, in which case we fall back to a single
    oversized page rather than dropping content, but 3-page groups of dense
    text were the actual cause of persistent 413s previously — a single
    dense page is enough context on its own.
    """
    chunks = []
    i = 0
    while i < len(pages):
        group = []
        size = 0
        while i < len(pages) and len(group) < pages_per_chunk:
            page_len = len(pages[i]["text"])
            # Always include at least one page per chunk, even if it alone
            # exceeds the cap (better to send one oversized request that
            # might need a retry than to silently truncate content).
            if group and size + page_len > max_chars_per_chunk:
                break
            group.append(pages[i])
            size += page_len
            i += 1

        combined_text = "\n\n".join(
            f"[PAGE {p['page_number']}]\n{p['text']}" for p in group
        )
        # Skip near-empty chunks (blank pages, pure images, etc.)
        if len(combined_text.strip()) < 40:
            continue
        chunks.append({
            "page_range": (group[0]["page_number"], group[-1]["page_number"]),
            "text": combined_text,
        })
    return chunks