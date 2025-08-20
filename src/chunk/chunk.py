from __future__ import annotations
import re
import hashlib
import logging
from collections import defaultdict, Counter
from typing import List, Tuple, Iterable, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    # Minimal config, to improve
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# ------------------------------------------------------------------------------
# Token-aware length (tiktoken) with fallback
# ------------------------------------------------------------------------------
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")

    def token_len(s: str) -> int:
        return len(_enc.encode(s))
except Exception:  # pragma: no cover
    def token_len(s: str) -> int:
        return len(s)

# ------------------------------------------------------------------------------
# Cleaning raw page text
# ------------------------------------------------------------------------------
def _clean_page_text(text: str) -> str:
    """
    Clean raw page text extracted from a PDF.
    - Remove hyphenation across line breaks.
    - Normalize multiple line breaks and spaces.
    - Strip extra whitespace.
    """
    text = re.sub(r'-\n(?=\w)', '', text)        # "inter-\naction" -> "interaction"
    text = re.sub(r'\r\n', '\n', text)           # normalize Windows line endings
    text = re.sub(r'[ \t]+\n', '\n', text)       # trailing spaces before newlines
    text = re.sub(r'\n{3,}', '\n\n', text)       # collapse multiple blank lines
    text = re.sub(r'[ \t]{2,}', ' ', text)       # collapse multiple spaces
    return text.strip()

# ------------------------------------------------------------------------------
# Header/Footer Detection
# ------------------------------------------------------------------------------
def _norm_repeat_line(line: str) -> str:
    """
    Normalize a single line for header/footer repetition detection:
    - trim + lowercase (casefold)
    - remove common page markers like "Page 12", "Pag. 3"
    - normalize inner whitespace
    - drop if it's only numbers
    """
    s = line.strip()
    # remove "Page 12" / "Pag. 3"
    s = re.sub(r'\b(?:page|pag\.?)\s*\d+\b', '', s, flags=re.I)
    # collapse whitespace
    s = re.sub(r'\s+', ' ', s)
    # pure numbers → empty
    if re.fullmatch(r'\d+', s):
        s = ''
    return s.casefold()

def _detect_repeat_lines(
    pages: List[Tuple[int, str]],
    top_k: int = 1,
    bottom_k: int = 1,
    min_ratio: float = 0.30,
):
    """
    Detect recurring headers and footers across multiple pages.
    - Looks at the top_k first lines and bottom_k last lines of each page.
    - Marks them as 'repeated' if they occur in more than `min_ratio` of pages.
    """
    tops, bottoms = Counter(), Counter()
    for _, txt in pages:
        lines = [l for l in txt.splitlines() if l.strip()]
        if not lines:
            continue
        # top
        for i in range(min(top_k, len(lines))):
            n = _norm_repeat_line(lines[i])
            if n:
                tops[n] += 1
        # bottom
        for i in range(1, min(bottom_k, len(lines)) + 1):
            n = _norm_repeat_line(lines[-i])
            if n:
                bottoms[n] += 1
    n_pages = max(1, len(pages))
    top_repeats = {t for t, c in tops.items() if c / n_pages >= min_ratio}
    bottom_repeats = {b for b, c in bottoms.items() if c / n_pages >= min_ratio}
    return top_repeats, bottom_repeats

def _strip_repeat_headers_footers(text: str, top_repeats, bottom_repeats) -> str:
    """Remove recurring header/footer lines from a single page of text."""
    lines = list(text.splitlines())
    # strip top
    while lines and _norm_repeat_line(lines[0]) in top_repeats:
        lines.pop(0)
    # strip bottom
    while lines and _norm_repeat_line(lines[-1]) in bottom_repeats:
        lines.pop()
    return "\n".join(lines)

# ------------------------------------------------------------------------------
# Chunking
# ------------------------------------------------------------------------------
def _make_chunk_id(doc_id: str, page: int, start: Optional[int], end: Optional[int], content: str) -> str:
    base = f"{doc_id}|{page}|{start}|{end}|{len(content)}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]

def chunk_pages(
    pages: List[Tuple[int, str]],
    pdf_path: str,
    doc_id: str | None = None,
    chunk_size_tokens: int = 700,
    chunk_overlap_tokens: int = 120,
) -> List[Document]:
    """
    Robust PDF text chunking:
      1) Clean text (hyphenation, whitespace).
      2) Remove recurring headers/footers (normalized).
      3) Create per-page Document objects with metadata.
      4) Split with token-aware RecursiveCharacterTextSplitter (regex separators, keep sep at end).
      5) Enrich chunk metadata: chunk_id (deterministic), token_count, start_char/end_char, chunk_index_page.
    """
    logger.info("Starting chunking process…")

    # Step 1: Clean text for each page
    cleaned: List[Tuple[int, str]] = [(p, _clean_page_text(t)) for p, t in pages]

    # Step 2: Remove recurring headers/footers
    top_rep, bottom_rep = _detect_repeat_lines(cleaned, top_k=1, bottom_k=1, min_ratio=0.30)
    stripped: List[Tuple[int, str]] = [(p, _strip_repeat_headers_footers(t, top_rep, bottom_rep)) for p, t in cleaned]

    # Step 3: Per-page Documents
    page_docs: List[Document] = []
    for page_num, page_text in stripped:
        if page_text.strip():
            page_docs.append(
                Document(
                    page_content=page_text,
                    metadata={
                        "source": pdf_path,
                        "page": int(page_num),
                        "doc_id": doc_id or pdf_path,
                        "type": "page",
                    },
                )
            )

    # Step 4: Token-aware splitting (paragraph → line → sentence-like → whitespace → char)
    # NOTE: from_tiktoken_encoder forwards kwargs to the underlying splitter.
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size_tokens,
        chunk_overlap=chunk_overlap_tokens,
        separators=[r"\n{2,}", r"\n", r"(?<=[\.!?])\s+", r"\s+", ""],
        is_separator_regex=True,
        keep_separator="end",          # keep delimiters at the end of each chunk
        add_start_index=True,          # include start index in metadata
        strip_whitespace=True,         # trim chunks
    )

    chunks = splitter.split_documents(page_docs)

    # Step 5: Enrich metadata with deterministic IDs and diagnostics
    out: List[Document] = []
    per_page_idx: Dict[int, int] = defaultdict(int)

    for d in chunks:
        meta: Dict[str, Any] = dict(d.metadata or {})
        pg = int(meta.get("page", -1))
        did = str(meta.get("doc_id", doc_id or pdf_path))

        # start / end char if available
        start_char: Optional[int] = meta.get("start_index")
        end_char: Optional[int] = (start_char + len(d.page_content)) if isinstance(start_char, int) else None

        # chunk index within the same page (stable ordering as produced)
        idx_in_page = per_page_idx[pg]
        per_page_idx[pg] += 1

        # token count for observability / cost
        tcount = token_len(d.page_content)

        # deterministic chunk id
        cid = _make_chunk_id(did, pg, start_char, end_char, d.page_content)

        meta.update({
            "chunk_id": cid,
            "type": "chunk",
            "token_count": tcount,
            "start_char": start_char,
            "end_char": end_char,
            "chunk_index_page": idx_in_page,
        })

        out.append(Document(page_content=d.page_content, metadata=meta))

    logger.info("Chunking complete. Created %d chunks.", len(out))
    return out
