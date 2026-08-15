"""
pipeline_runner.py

Enterprise RAG Pipeline Orchestrator
------------------------------------

Pipeline:

    1. Gmail ingestion
    2. Data cleaning
    3. Incremental file filtering
    4. Document processing
    5. Text cleaning
    6. Chunking
    7. Embedding
    8. Chroma vector store
    9. Update processing tracker
    10. Update last_sync.txt

INCREMENTAL PROCESSING
----------------------

Gmail ingestion is controlled by:

    last_sync.txt

Example:

    2026/08/13

The Gmail ingestion module searches only from the
last synchronization window.

Document processing is controlled by:

    data/processed_files.json

This prevents files that have already been successfully
embedded and stored from being processed again.

IMPORTANT
---------

This script NEVER deletes:

    - Chroma database
    - processed_files.json
    - token.json
    - last_sync.txt
    - downloaded Gmail attachments

The existing vector store therefore remains intact.

last_sync.txt is only ever updated AFTER Gmail ingestion has
succeeded for this run (whether or not there were new documents
to process). It is never updated if Gmail ingestion itself fails.
"""

import os
import sys
import json
from pathlib import Path
from datetime import date


# ==========================================================================
# PATH SETUP
# ==========================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PIPELINES_DIR = PROJECT_ROOT / "pipelines"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"


sys.path.append(str(PIPELINES_DIR))
sys.path.append(str(VECTORSTORE_DIR))


# ==========================================================================
# IMPORTS
# ==========================================================================

# Gmail ingestion
from gmail_ingestion import main as run_gmail_ingestion


# Data cleaning
from data_cleaning import run_data_cleaning_pipeline


# Document processing
from document_processing import run_document_processing_pipeline


# Text cleaning
from text_cleaning import clean_document


# Chunking
from chunking import chunk_document


# Embedding
from embedding import embed_chunks


# Vector store
from vector_store import get_vector_store


# ==========================================================================
# CONFIGURATION
# ==========================================================================

# File that tracks successfully processed files
TRACKING_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed_files.json"
)


# Chroma database location
CHROMA_PERSIST_DIR = str(
    PROJECT_ROOT
    / "vectorstore"
    / "chroma_db"
)


# Chroma collection
COLLECTION_NAME = "rag_documents"


# last_sync.txt location
LAST_SYNC_FILE = PROJECT_ROOT / "last_sync.txt"


# ==========================================================================
# FILE TRACKING
# ==========================================================================

def load_processed_file_ids() -> set:
    """
    Load the list of files that have already been
    successfully processed.

    Returns:
        set of absolute file paths
    """

    if not TRACKING_FILE.exists():

        print(
            f"[INFO] Tracking file does not exist yet:\n"
            f"       {TRACKING_FILE}"
        )

        return set()

    try:

        data = json.loads(
            TRACKING_FILE.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(data, list):

            print(
                "[WARN] processed_files.json does not "
                "contain a list."
            )

            return set()

        return set(data)

    except json.JSONDecodeError:

        print(
            f"[WARN] Could not parse:\n"
            f"       {TRACKING_FILE}"
        )

        print(
            "[WARN] Existing tracking file will NOT "
            "be overwritten automatically."
        )

        return set()

    except OSError as e:

        print(
            f"[WARN] Could not read tracking file: {e}"
        )

        return set()


def save_processed_file_ids(
    processed_ids: set
) -> None:
    """
    Save successfully processed file paths.
    """

    TRACKING_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    TRACKING_FILE.write_text(
        json.dumps(
            sorted(processed_ids),
            indent=2
        ),
        encoding="utf-8"
    )


def file_id(file_path) -> str:
    """
    Convert a file path into a stable absolute path.
    """

    return str(
        Path(file_path).resolve()
    )


def update_last_sync() -> None:
    """
    Advance last_sync.txt to today's date.

    Only ever called after Gmail ingestion has succeeded
    for this run -- whether or not there were new documents
    to process afterward. Never called if Gmail ingestion
    itself failed.
    """

    today_str = date.today().strftime("%Y/%m/%d")

    LAST_SYNC_FILE.write_text(
        today_str,
        encoding="utf-8"
    )

    print(
        f"Last sync date updated to: {today_str}"
    )


# ==========================================================================
# MAIN PIPELINE
# ==========================================================================

def main():

    print()
    print("=" * 70)
    print("        ENTERPRISE RAG INGESTION PIPELINE")
    print("=" * 70)
    print()

    print(
        f"Project root : {PROJECT_ROOT}"
    )

    print(
        f"Tracking file: {TRACKING_FILE}"
    )

    print(
        f"Chroma store : {CHROMA_PERSIST_DIR}"
    )

    print(
        f"Collection   : {COLLECTION_NAME}"
    )

    print()


    # ======================================================================
    # STEP 1 — GMAIL INGESTION
    # ======================================================================

    print(
        "=" * 70
    )

    print(
        "STEP 1: GMAIL INGESTION"
    )

    print(
        "=" * 70
    )

    print()

    try:

        run_gmail_ingestion()

    except Exception as e:

        print()
        print(
            "[ERROR] Gmail ingestion failed."
        )

        print(
            f"Reason: {e}"
        )

        print()
        print(
            "Pipeline stopped."
        )

        print(
            "[IMPORTANT] last_sync.txt was NOT updated."
        )

        return


    print()

    print(
        "[OK] Gmail ingestion completed."
    )

    print()


    # ======================================================================
    # STEP 2 — LOAD PROCESSED FILE TRACKING
    # ======================================================================

    print(
        "=" * 70
    )

    print(
        "STEP 2: LOAD PROCESSING TRACKER"
    )

    print(
        "=" * 70
    )

    print()

    processed_ids = load_processed_file_ids()

    print(
        f"Previously processed files: "
        f"{len(processed_ids)}"
    )

    print()


    # ======================================================================
    # STEP 3 — DATA CLEANING
    # ======================================================================

    print(
        "=" * 70
    )

    print(
        "STEP 3: DATA CLEANING"
    )

    print(
        "=" * 70
    )

    print()

    try:

        cleaned_documents = (
            run_data_cleaning_pipeline(processed_ids)
        )

    except Exception as e:

        print()

        print(
            "[ERROR] Data cleaning failed."
        )

        print(
            f"Reason: {e}"
        )

        return


    print()


    # ======================================================================
    # STEP 4 — INCREMENTAL FILE FILTER
    # ======================================================================

    print(
        "=" * 70
    )

    print(
        "STEP 4: INCREMENTAL FILE FILTER"
    )

    print(
        "=" * 70
    )

    print()


    new_documents = []

    already_processed = []


    for file_path in cleaned_documents:

        current_id = file_id(file_path)

        if current_id in processed_ids:

            already_processed.append(
                file_path
            )

        else:

            new_documents.append(
                file_path
            )


    print(
        f"Total valid files after cleaning : "
        f"{len(cleaned_documents)}"
    )

    print(
        f"Already processed                 : "
        f"{len(already_processed)}"
    )

    print(
        f"New files to process              : "
        f"{len(new_documents)}"
    )

    print()


    # ======================================================================
    # NOTHING NEW
    # ======================================================================

    if not new_documents:

        print(
            "=" * 70
        )

        print(
            "NO NEW DOCUMENTS"
        )

        print(
            "=" * 70
        )

        print()

        print(
            "All valid files have already been "
            "processed and stored."
        )

        print()

        # Gmail ingestion succeeded this run (Step 1 completed),
        # even though there was nothing new to embed. Advance
        # last_sync.txt so the search window doesn't keep growing
        # on every subsequent run.
        update_last_sync()

        return


    # ======================================================================
    # STEP 5 — DOCUMENT PROCESSING
    # ======================================================================

    print(
        "=" * 70
    )

    print(
        "STEP 5: DOCUMENT PROCESSING"
    )

    print(
        "=" * 70
    )

    print()

    try:

        processed_documents = (
            run_document_processing_pipeline(
                new_documents
            )
        )

    except Exception as e:

        print()

        print(
            "[ERROR] Document processing failed."
        )

        print(
            f"Reason: {e}"
        )

        return


    print()


    # ======================================================================
    # STEP 6 — TEXT CLEANING + CHUNKING
    # ======================================================================

    print(
        "=" * 70
    )

    print(
        "STEP 6: TEXT CLEANING + CHUNKING"
    )

    print(
        "=" * 70
    )

    print()


    all_chunks = []

    skipped_empty = 0

    successfully_processed_ids = set()


    for document in processed_documents:

        # --------------------------------------------------------------
        # Get raw extracted text
        # --------------------------------------------------------------

        raw_text = document.get(
            "text",
            ""
        )


        # --------------------------------------------------------------
        # Get source path
        # --------------------------------------------------------------

        source_path = document.get(
            "file_path"
        )


        if not source_path:

            print(
                "[WARN] Document has no file_path."
            )

            skipped_empty += 1

            continue


        # --------------------------------------------------------------
        # Empty raw text
        # --------------------------------------------------------------

        if not raw_text or not raw_text.strip():

            skipped_empty += 1

            print(
                f"[skip] Empty text: "
                f"{source_path}"
            )

            continue


        # --------------------------------------------------------------
        # Text cleaning
        # --------------------------------------------------------------

        cleaned_text = clean_document(
            raw_text
        )


        if not cleaned_text or not cleaned_text.strip():

            skipped_empty += 1

            print(
                f"[skip] Empty after cleaning: "
                f"{source_path}"
            )

            continue


        # --------------------------------------------------------------
        # Source filename
        # --------------------------------------------------------------

        metadata = document.get(
            "metadata",
            {}
        )


        source_name = metadata.get(
            "file_name"
        )


        if not source_name:

            source_name = Path(
                source_path
            ).name


        # --------------------------------------------------------------
        # Chunk document
        # --------------------------------------------------------------

        chunks = chunk_document(

            cleaned_text,

            strategy="recursive",

            max_chars=1000,

            overlap=100,

            source=source_name,
        )


        if not chunks:

            skipped_empty += 1

            print(
                f"[skip] No chunks created: "
                f"{source_name}"
            )

            continue


        all_chunks.extend(
            chunks
        )


        # --------------------------------------------------------------
        # Mark only successfully chunked documents
        # --------------------------------------------------------------

        successfully_processed_ids.add(
            file_id(source_path)
        )


    # ======================================================================
    # CHUNKING REPORT
    # ======================================================================

    print()

    print(
        "=" * 70
    )

    print(
        "CHUNKING REPORT"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"Documents skipped : "
        f"{skipped_empty}"
    )

    print(
        f"Total chunks      : "
        f"{len(all_chunks)}"
    )

    print()


    # ======================================================================
    # NO CHUNKS
    # ======================================================================

    if not all_chunks:

        print(
            "[WARN] No chunks available for embedding."
        )

        print(
            "[WARN] processed_files.json will NOT "
            "be updated."
        )

        # Gmail ingestion still succeeded this run, so we still
        # advance last_sync.txt even though nothing got embedded.
        update_last_sync()

        return


    # ======================================================================
    # STEP 7 — EMBEDDING
    # ======================================================================

    print(
        "=" * 70
    )

    print(
        "STEP 7: EMBEDDING"
    )

    print(
        "=" * 70
    )

    print()


    try:

        embedded_chunks = embed_chunks(

            all_chunks,

            backend="sentence_transformers",
        )

    except Exception as e:

        print()

        print(
            "[ERROR] Embedding failed."
        )

        print(
            f"Reason: {e}"
        )

        print()

        print(
            "[IMPORTANT] processed_files.json "
            "was NOT updated."
        )

        print(
            "[IMPORTANT] last_sync.txt was NOT updated."
        )

        return


    print()

    print(
        f"Embeddings created: "
        f"{len(embedded_chunks)}"
    )

    print()


    # ======================================================================
    # STEP 8 — CHROMA VECTOR STORE
    # ======================================================================

    print(
        "=" * 70
    )

    print(
        "STEP 8: CHROMA VECTOR STORE"
    )

    print(
        "=" * 70
    )

    print()


    try:

        store = get_vector_store(

            backend="chroma",

            collection_name=COLLECTION_NAME,

            persist_dir=CHROMA_PERSIST_DIR,
        )


        # --------------------------------------------------------------
        # IMPORTANT:
        #
        # add_chunks() adds the new chunks to the existing collection.
        #
        # It does NOT delete the existing Chroma database.
        # --------------------------------------------------------------

        store.add_chunks(
            embedded_chunks
        )


    except Exception as e:

        print()

        print(
            "[ERROR] Failed to store embeddings "
            "in Chroma."
        )

        print(
            f"Reason: {e}"
        )

        print()

        print(
            "[IMPORTANT] processed_files.json "
            "was NOT updated."
        )

        print(
            "[IMPORTANT] last_sync.txt was NOT updated."
        )

        return


    # ======================================================================
    # VECTOR STORE REPORT
    # ======================================================================

    print()

    print(
        "=" * 70
    )

    print(
        "VECTOR STORE REPORT"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"Total chunks in Chroma: "
        f"{store.count()}"
    )

    print()


    # ======================================================================
    # STEP 9 — UPDATE PROCESSING TRACKER
    # ======================================================================

    print(
        "=" * 70
    )

    print(
        "STEP 9: UPDATE PROCESSING TRACKER"
    )

    print(
        "=" * 70
    )

    print()


    # Only update the tracker AFTER Chroma storage succeeds.

    processed_ids.update(
        successfully_processed_ids
    )


    save_processed_file_ids(
        processed_ids
    )


    print(
        f"Successfully marked "
        f"{len(successfully_processed_ids)} "
        f"new file(s) as processed."
    )

    print(
        f"Total tracked files: "
        f"{len(processed_ids)}"
    )

    print()


    # ======================================================================
    # STEP 10 — UPDATE LAST SYNC DATE
    # ======================================================================

    print(
        "=" * 70
    )

    print(
        "STEP 10: UPDATE LAST SYNC DATE"
    )

    print(
        "=" * 70
    )

    print()

    # Only reached after the full pipeline (Gmail -> cleaning ->
    # processing -> chunking -> embedding -> Chroma -> tracker)
    # has succeeded end to end.

    update_last_sync()

    print()


    # ======================================================================
    # FINAL
    # ======================================================================

    print(
        "=" * 70
    )

    print(
        "RAG PIPELINE COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"New documents processed : "
        f"{len(successfully_processed_ids)}"
    )

    print(
        f"New chunks stored       : "
        f"{len(embedded_chunks)}"
    )

    print(
        f"Total Chroma chunks     : "
        f"{store.count()}"
    )

    print()

    print(
        "Existing vector-store data was preserved."
    )

    print(
        "Existing processed-file tracking was preserved."
    )

    print()


# ==========================================================================
# ENTRY POINT
# ==========================================================================

if __name__ == "__main__":
    main()