from pathlib import Path
from datetime import datetime

from docx import Document
import fitz  # PyMuPDF


# =====================================================
# Document Loading
# =====================================================

def load_documents(documents):

    loaded_documents = []

    for file in documents:

        loaded_documents.append(
            {
                "file_path": file
            }
        )

    print("\n========== Document Loading Report ==========")
    print(f"Documents Loaded : {len(loaded_documents)}")
    print("=============================================")

    return loaded_documents


# =====================================================
# Metadata Extraction
# =====================================================

def extract_metadata(documents):

    processed_documents = []

    for document in documents:

        file = document["file_path"]

        metadata = {

            "file_name": file.name,

            "file_type": file.suffix.lower(),

            "file_size_kb": round(
                file.stat().st_size / 1024,
                2
            ),

            "created_date": datetime.fromtimestamp(
                file.stat().st_ctime
            ).strftime("%Y-%m-%d %H:%M:%S"),

            "modified_date": datetime.fromtimestamp(
                file.stat().st_mtime
            ).strftime("%Y-%m-%d %H:%M:%S")

        }

        document["metadata"] = metadata

        processed_documents.append(document)

    print("\n========== Metadata Extraction Report ==========")
    print(f"Metadata Extracted : {len(processed_documents)}")
    print("================================================")

    return processed_documents


# =====================================================
# PDF Text Extraction (word-token based, via PyMuPDF)
# =====================================================
#
# Why word-token extraction instead of page.get_text("text"):
#
# PyMuPDF's plain "text" mode reconstructs words by measuring gaps
# between characters. On PDFs with unusual letter/word spacing this
# guess can go wrong in BOTH directions:
#   - gaps get treated as word breaks that aren't real  -> "C om pan y"
#   - real word breaks get missed because the gap is too small -> "andlevies"
#
# "words" mode sidesteps this entirely: PyMuPDF's layout engine has
# already tokenized the page into discrete words (with bounding boxes,
# block number, line number, and word-in-line index) during PDF parsing.
# We just join those tokens back together with a single explicit space,
# so word boundaries are never guessed from character spacing again.

def _extract_pdf_text(file):

    text_parts = []

    pdf = fitz.open(file)

    for page in pdf:

        words = page.get_text("words")

        if not words:
            continue

        # Each word tuple: (x0, y0, x1, y1, "word", block_no, line_no, word_no)
        # Sort into natural reading order: block -> line -> word position
        words.sort(key=lambda w: (w[5], w[6], w[7]))

        lines = []
        current_line_key = None
        current_line_words = []

        for word in words:

            line_key = (word[5], word[6])  # (block_no, line_no)
            word_text = word[4]

            if line_key != current_line_key:

                if current_line_words:
                    lines.append(" ".join(current_line_words))

                current_line_words = [word_text]
                current_line_key = line_key

            else:
                current_line_words.append(word_text)

        if current_line_words:
            lines.append(" ".join(current_line_words))

        text_parts.append("\n".join(lines))

    pdf.close()

    return "\n".join(text_parts)


# =====================================================
# Text Extraction
# =====================================================

def extract_text(documents):

    processed_documents = []

    for document in documents:

        file = document["file_path"]

        extracted_text = ""

        extension = file.suffix.lower()

        # PDF Extraction (PyMuPDF, word-token based)
        if extension == ".pdf":

            extracted_text = _extract_pdf_text(file)

        # DOCX Extraction
        elif extension == ".docx":

            doc = Document(file)

            for paragraph in doc.paragraphs:

                extracted_text += paragraph.text + "\n"

        # TXT Extraction
        elif extension == ".txt":

            with open(
                file,
                "r",
                encoding="utf-8",
                errors="ignore"
            ) as f:

                extracted_text = f.read()

        document["text"] = extracted_text

        processed_documents.append(document)

    print("\n========== Text Extraction Report ==========")
    print(f"Text Extracted : {len(processed_documents)}")
    print("============================================")

    return processed_documents


# =====================================================
# Document Processing Pipeline
# =====================================================

def run_document_processing_pipeline(documents):

    documents = load_documents(documents)

    documents = extract_metadata(documents)

    documents = extract_text(documents)

    return documents


# =====================================================
# Standalone Test Runner
# =====================================================

if __name__ == "__main__":
    from data_cleaning import run_data_cleaning_pipeline

    cleaned_documents = run_data_cleaning_pipeline()
    run_document_processing_pipeline(cleaned_documents)