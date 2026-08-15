"""
text_cleaning.py

Full text cleaning pipeline for Generative AI / RAG document processing.
Handles: whitespace normalization, page numbers, headers/footers,
unicode normalization, control characters, duplicate lines, broken
letter-spacing (from certain PDF extractions), and special character
stripping (while preserving script/multilingual text).
"""

import re
import unicodedata


# ---------------------------------------------------------------------------
# Step 1: Unicode normalization + smart-quote / dash cleanup
# ---------------------------------------------------------------------------
def normalize_unicode(text: str) -> str:
    """
    Normalize unicode form (NFKC) and replace common 'smart' typographic
    characters (curly quotes, em/en dashes, ellipsis) with plain ASCII
    equivalents. This matters a lot for embeddings/tokenizers trained
    mostly on plain text.
    """
    text = unicodedata.normalize("NFKC", text)

    replacements = {
        "\u2018": "'", "\u2019": "'",   # ' '
        "\u201c": '"', "\u201d": '"',   # " "
        "\u2013": "-", "\u2014": "-",   # – —
        "\u2026": "...",                # …
        "\xa0": " ",                    # non-breaking space
    }
    for src, tgt in replacements.items():
        text = text.replace(src, tgt)

    return text


# ---------------------------------------------------------------------------
# Step 2: Fix broken letter-spacing from certain PDF extractions
# ---------------------------------------------------------------------------
def fix_broken_spacing(text: str) -> str:
    """
    Some PDFs (often ones with justified/stretched text or unusual font
    encoding) extract with stray spaces jammed inside words, e.g.:
        "C om pan y"  ->  should be "Company"
        "In tellectu al Pro per ty" -> "Intellectual Property"

    This happens because pypdf reconstructs text based on character
    x-position on the page, and wide letter-spacing gets misread as
    word boundaries.

    Heuristic: find runs of 3+ consecutive short (1-3 letter) alpha
    fragments separated by single spaces, and join them into one word.
    This is a best-effort fix, not perfect -- it can occasionally merge
    short real words (e.g. "a an in") if they appear consecutively, so
    it requires a run of at least 3 fragments to reduce false positives.
    """
    def merge(match):
        return match.group(0).replace(" ", "")

    # Matches: 3+ short alpha fragments (1-3 chars) chained by single spaces
    pattern = re.compile(r'\b(?:[A-Za-z]{1,3}\s){2,}[A-Za-z]{1,3}\b')
    return pattern.sub(merge, text)


# ---------------------------------------------------------------------------
# Step 3: Strip control / non-printable characters
# ---------------------------------------------------------------------------
def remove_control_characters(text: str) -> str:
    """
    Remove non-printable / control characters that commonly leak in from
    PDF extraction (e.g. \\x00, \\x0c form-feed), but keep normal
    whitespace (space, tab, newline).
    """
    return "".join(
        ch for ch in text
        if ch in ("\n", "\t", " ") or unicodedata.category(ch)[0] != "C"
    )


# ---------------------------------------------------------------------------
# Step 4: Remove header/footer/page-number lines (line-by-line, BEFORE
# whitespace is collapsed — this avoids the classic bug of `.*` eating
# the rest of the document once newlines are gone).
# ---------------------------------------------------------------------------
def remove_header_footer_lines(text: str) -> str:
    """
    NOTE: a previous version of this function also matched a bare lone
    number on its own line (r'^\\s*\\d+\\s*$') as a "page number". That
    pattern is too broad for tabular/financial documents: PDF word-token
    extraction puts each table cell on its own line, so line-item costs,
    quantities, and totals (e.g. "630", "4853") are indistinguishable
    from page numbers under that regex and were being silently deleted
    before chunking/embedding ever ran. Removed that pattern -- only the
    unambiguous "Page N" / "Page N of M" form is stripped now.
    """
    patterns = [
        r'^\s*Page\s*\d+(\s*of\s*\d+)?\s*$',   # Page 1 / Page 1 of 10
        r'.*Copyright.*',
        r'.*Confidential.*',
        r'.*All rights reserved.*',
    ]
    combined = re.compile("|".join(f"(?:{p})" for p in patterns),
                           flags=re.IGNORECASE | re.MULTILINE)
    return combined.sub('', text)


# ---------------------------------------------------------------------------
# Step 5: Remove duplicate lines (repeated headers/footers across pages)
# ---------------------------------------------------------------------------
def remove_duplicate_lines(text: str, min_repeats: int = 1) -> str:
    """
    Remove lines that repeat verbatim.

    min_repeats: a line must appear MORE than this many times to be
    treated as a repeated header/footer and collapsed to a single
    occurrence. Use min_repeats > 1 if you're worried about removing
    genuinely repeated short content (e.g. "N/A") that only appears
    once or twice.
    """
    lines = text.split("\n")
    counts = {}
    for line in lines:
        stripped = line.strip()
        if stripped:
            counts[stripped] = counts.get(stripped, 0) + 1

    seen = set()
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if counts[stripped] > min_repeats:
            # repeated line -> keep only the first occurrence
            if stripped not in seen:
                cleaned_lines.append(stripped)
                seen.add(stripped)
        else:
            cleaned_lines.append(stripped)

    return "\n".join(cleaned_lines)


# ---------------------------------------------------------------------------
# Step 6: Core character-level cleaning + whitespace collapse
# ---------------------------------------------------------------------------
def clean_text(text: str, keep_chars: str = r".,:;!?()\-₹$%@/&'\"") -> str:
    """
    Collapse whitespace and strip characters that aren't word characters,
    normal whitespace, or explicitly whitelisted punctuation/symbols.

    `\\w` matches unicode word characters by default in Python's `re`,
    so non-English scripts (Hindi, etc.) are preserved automatically.
    """
    # Remove everything not in: word chars, whitespace, or keep_chars
    pattern = rf'[^\w\s{keep_chars}]'
    text = re.sub(pattern, '', text)

    # Collapse all whitespace (spaces, tabs, newlines) into single spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def clean_document(text: str, min_repeats: int = 1) -> str:
    """
    Complete text cleaning pipeline, in the correct order:
      1. Unicode normalization
      2. Fix broken letter-spacing (PDF extraction artifact)
      3. Remove control characters
      4. Remove header/footer/page-number lines (line-based, pre-collapse)
      5. Remove duplicate lines
      6. Character-level cleanup + whitespace collapse
    """
    text = normalize_unicode(text)
    text = fix_broken_spacing(text)
    text = remove_control_characters(text)
    text = remove_header_footer_lines(text)
    text = remove_duplicate_lines(text, min_repeats=min_repeats)
    text = clean_text(text)

    return text


if __name__ == "__main__":

    sample_text = """
    Page 1

    ANNUAL REPORT 2024

    Copyright © 2024 Acme Corp. All rights reserved.

    This section explains our revenue growth of 25% in FY24,
    driven by strong demand in the ₹ and $ markets.

    The loss occ asione d to the C om pan y shal l be dedu cted.
    In tellectu al Pro per ty R ig h ts are gov erned by this clause.

    ANNUAL REPORT 2024

    Page 2

    Contact us at support@acme.com for more details.

    Copyright © 2024 Acme Corp. All rights reserved.
    """

    cleaned = clean_document(sample_text)
    print("--- CLEANED OUTPUT ---")
    print(cleaned)