"""
chunking.py

Text chunking pipeline for RAG document processing.
Takes cleaned text (output of text_cleaning.py) and splits it into
retrieval-friendly chunks.

Strategies included:
  1. fixed_size_chunk        - simple character-based chunking with overlap
  2. sentence_chunk          - groups whole sentences up to a max size
  3. recursive_chunk         - splits by paragraph -> sentence -> word,
                                 falling back progressively (LangChain-style)
  4. token_based_chunk       - chunk by token count (uses tiktoken if
                                 available, falls back to word-count estimate)

Each chunk is returned as a dict with metadata (id, text, char/word count,
start/end offsets) so it can be traced back to the source document later.

NOTE: chunk_document() stamps every chunk's TEXT (not just its metadata)
with a "[Source: ...]" label before returning it. This matters because
retrieval (both BM25 and embeddings) only ever sees the "text" field --
metadata like "source" is never searched. Without the stamp, a chunk that
doesn't happen to contain any identifying words (e.g. a part number that
got split into an earlier chunk) is invisible to queries about that
document, and can be indistinguishable from a near-duplicate chunk in a
different document that uses similar boilerplate wording.
"""

import re
import uuid
import os
from typing import List, Dict, Optional


# ---------------------------------------------------------------------------
# Helper: split text into sentences (simple, dependency-free)
# ---------------------------------------------------------------------------
def split_into_sentences(text: str) -> List[str]:
    """
    Lightweight sentence splitter using punctuation boundaries.
    Not as accurate as spaCy/nltk, but has zero dependencies.
    For higher accuracy on messy/domain text, swap in nltk.sent_tokenize
    or spaCy's sentencizer.
    """
    sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])')
    sentences = sentence_endings.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


# ---------------------------------------------------------------------------
# Helper: split text into paragraphs
# ---------------------------------------------------------------------------
def split_into_paragraphs(text: str) -> List[str]:
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paragraphs if p.strip()]


# ---------------------------------------------------------------------------
# Helper: build a chunk dict with consistent metadata
# ---------------------------------------------------------------------------
def _make_chunk(text: str, index: int, start: Optional[int] = None,
                 end: Optional[int] = None, source: Optional[str] = None) -> Dict:
    return {
        "chunk_id": str(uuid.uuid4()),
        "index": index,
        "text": text,
        "char_count": len(text),
        "word_count": len(text.split()),
        "start_offset": start,
        "end_offset": end,
        "source": source,
    }


# ---------------------------------------------------------------------------
# Helper: derive a short, human-readable label from a source path/filename
# ---------------------------------------------------------------------------
def _source_label(source: str) -> str:
    """
    Turns a path like '.../Quotation for Part-001 - Sample Component.pdf'
    into 'Quotation for Part-001 - Sample Component' -- strips directory
    and extension so the stamp reads cleanly and still carries every
    identifying token (part numbers, customer names, etc.) that appears in
    the filename.
    """
    base = os.path.basename(source)
    name, _ext = os.path.splitext(base)
    return name.strip()


# ---------------------------------------------------------------------------
# Helper: stamp a chunk's TEXT with its source label
# ---------------------------------------------------------------------------
def _stamp_chunk_with_source(chunk: Dict) -> Dict:
    """
    Prepends a '[Source: <label>]' line to chunk['text'] and recomputes the
    derived char/word counts. Applied uniformly to the output of every
    strategy inside chunk_document(), so no chunk -- regardless of which
    piece of the document it happens to contain -- ever loses its identity
    once it's split off on its own.
    """
    source = chunk.get("source")
    if not source:
        return chunk

    label = _source_label(source)
    if not label:
        return chunk

    stamped_text = f"[Source: {label}]\n{chunk['text']}"
    chunk["text"] = stamped_text
    chunk["char_count"] = len(stamped_text)
    chunk["word_count"] = len(stamped_text.split())
    return chunk


# ---------------------------------------------------------------------------
# Strategy 1: Fixed-size chunking with overlap
# ---------------------------------------------------------------------------
def fixed_size_chunk(text: str, chunk_size: int = 1000, overlap: int = 150,
                      source: Optional[str] = None) -> List[Dict]:
    """
    Split text into fixed-size character chunks with overlap between
    consecutive chunks (helps preserve context across chunk boundaries).

    chunk_size: max characters per chunk
    overlap: characters repeated from the end of the previous chunk
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    index = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(_make_chunk(chunk_text, index, start, end, source))
            index += 1

        if end == text_len:
            break

        start = end - overlap  # step forward, keeping overlap

    return chunks


# ---------------------------------------------------------------------------
# Strategy 2: Sentence-aware chunking
# ---------------------------------------------------------------------------
def sentence_chunk(text: str, max_chars: int = 1000, overlap_sentences: int = 1,
                    source: Optional[str] = None) -> List[Dict]:
    """
    Groups whole sentences together until max_chars is reached, so chunks
    never cut a sentence in half. Overlaps the last N sentences into the
    next chunk for context continuity.
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    chunks = []
    current: List[str] = []
    current_len = 0
    index = 0

    for sentence in sentences:
        sentence_len = len(sentence) + 1

        if current and current_len + sentence_len > max_chars:
            chunk_text = " ".join(current)
            chunks.append(_make_chunk(chunk_text, index, source=source))
            index += 1

            # carry forward overlap sentences for context
            current = current[-overlap_sentences:] if overlap_sentences > 0 else []
            current_len = sum(len(s) + 1 for s in current)

        current.append(sentence)
        current_len += sentence_len

    if current:
        chunks.append(_make_chunk(" ".join(current), index, source=source))

    return chunks


# ---------------------------------------------------------------------------
# Strategy 3: Recursive chunking (paragraph -> sentence -> word fallback)
# ---------------------------------------------------------------------------
def recursive_chunk(text: str, max_chars: int = 1000, overlap: int = 100,
                     source: Optional[str] = None) -> List[Dict]:
    """
    LangChain-style recursive splitting. Tries to keep paragraphs whole;
    if a paragraph is too big, falls back to splitting it by sentence;
    if a sentence is still too big, falls back to a hard character split.
    This gives the best balance of semantic coherence and size control.
    """
    paragraphs = split_into_paragraphs(text)
    raw_pieces: List[str] = []

    for para in paragraphs:
        if len(para) <= max_chars:
            raw_pieces.append(para)
            continue

        # paragraph too big -> split by sentence
        sentences = split_into_sentences(para)
        buffer = ""
        for sentence in sentences:
            if len(sentence) > max_chars:
                # sentence itself too big -> hard split by characters
                if buffer:
                    raw_pieces.append(buffer)
                    buffer = ""
                for i in range(0, len(sentence), max_chars):
                    raw_pieces.append(sentence[i:i + max_chars])
                continue

            if len(buffer) + len(sentence) + 1 <= max_chars:
                buffer = f"{buffer} {sentence}".strip()
            else:
                raw_pieces.append(buffer)
                buffer = sentence

        if buffer:
            raw_pieces.append(buffer)

    # merge small adjacent pieces up to max_chars, adding overlap
    chunks = []
    index = 0
    i = 0
    while i < len(raw_pieces):
        piece = raw_pieces[i]
        merged = piece
        j = i + 1
        while j < len(raw_pieces) and len(merged) + len(raw_pieces[j]) + 1 <= max_chars:
            merged = f"{merged} {raw_pieces[j]}".strip()
            j += 1

        # add trailing overlap from the merged text's own tail into next start
        chunks.append(_make_chunk(merged, index, source=source))
        index += 1
        i = j if j > i else i + 1

    # apply simple overlap by prepending tail of previous chunk
    if overlap > 0:
        for k in range(1, len(chunks)):
            tail = chunks[k - 1]["text"][-overlap:]
            chunks[k]["text"] = f"{tail} {chunks[k]['text']}".strip()
            chunks[k]["char_count"] = len(chunks[k]["text"])
            chunks[k]["word_count"] = len(chunks[k]["text"].split())

    return chunks


# ---------------------------------------------------------------------------
# Strategy 4: Token-based chunking (best for LLM context-window budgeting)
# ---------------------------------------------------------------------------
def token_based_chunk(text: str, max_tokens: int = 300, overlap_tokens: int = 40,
                       model_name: str = "cl100k_base",
                       source: Optional[str] = None) -> List[Dict]:
    """
    Chunk by token count rather than characters -- important because LLM
    context limits and embedding-model limits are token-based, not
    character-based.

    Uses `tiktoken` if installed (pip install tiktoken) for exact counts.
    Falls back to a ~4-chars-per-token estimate if tiktoken isn't available.
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding(model_name)
        tokens = enc.encode(text)

        chunks = []
        start = 0
        index = 0
        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = enc.decode(chunk_tokens).strip()

            if chunk_text:
                chunk = _make_chunk(chunk_text, index, source=source)
                chunk["token_count"] = len(chunk_tokens)
                chunks.append(chunk)
                index += 1

            if end == len(tokens):
                break
            start = end - overlap_tokens

        return chunks

    except ImportError:
        # fallback: approximate 1 token ~= 4 characters (English average)
        approx_chunk_chars = max_tokens * 4
        approx_overlap_chars = overlap_tokens * 4
        chunks = fixed_size_chunk(text, chunk_size=approx_chunk_chars,
                                   overlap=approx_overlap_chars, source=source)
        for c in chunks:
            c["token_count_estimated"] = round(c["char_count"] / 4)
        return chunks


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def chunk_document(text: str, strategy: str = "recursive", **kwargs) -> List[Dict]:
    """
    Single entry point for the pipeline. `strategy` selects the method:
      "fixed"     -> fixed_size_chunk
      "sentence"  -> sentence_chunk
      "recursive" -> recursive_chunk   (recommended default)
      "token"     -> token_based_chunk

    After the chosen strategy produces its chunks, every chunk's TEXT is
    stamped with a "[Source: <filename>]" line (see _stamp_chunk_with_source).
    This runs regardless of strategy, so the fix applies uniformly whichever
    of the 4 methods is in use -- and regardless of which arbitrary slice
    of the document a given chunk happens to contain.
    """
    strategies = {
        "fixed": fixed_size_chunk,
        "sentence": sentence_chunk,
        "recursive": recursive_chunk,
        "token": token_based_chunk,
    }

    if strategy not in strategies:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from {list(strategies)}")

    chunks = strategies[strategy](text, **kwargs)
    chunks = [_stamp_chunk_with_source(c) for c in chunks]
    return chunks


if __name__ == "__main__":

    sample_text = """
    Retrieval-Augmented Generation (RAG) combines a retriever with a generator.
    The retriever fetches relevant chunks of text from a knowledge base. The
    generator then uses those chunks as context to produce an accurate answer.

    Chunking strategy matters a lot for retrieval quality. Chunks that are too
    large dilute relevance during similarity search. Chunks that are too small
    lose surrounding context needed for the LLM to answer correctly.

    A good default is 500 to 1000 characters per chunk, with 10 to 20 percent
    overlap between consecutive chunks to preserve context across boundaries.
    """

    print("=== RECURSIVE CHUNKING (with source stamping) ===")
    for c in chunk_document(sample_text, strategy="recursive", max_chars=300,
                             overlap=40, source="sample_doc.txt"):
        print(f"[{c['index']}] ({c['char_count']} chars) {c['text']}\n")

    print("=== SENTENCE CHUNKING (with source stamping) ===")
    for c in chunk_document(sample_text, strategy="sentence", max_chars=300,
                             source="sample_doc.txt"):
        print(f"[{c['index']}] ({c['char_count']} chars) {c['text']}\n")
