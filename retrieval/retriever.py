"""
retriever.py

Advanced Retrieval Layer for RAG

Techniques:
    1. Semantic Search
    2. BM25 Keyword Search
    3. Hybrid Retrieval
    4. Reciprocal Rank Fusion (RRF)

Important:
    Semantic Search and BM25 use the SAME chunk_id.
    This allows RRF to correctly combine results from
    both retrieval methods.

FIXES APPLIED (see chat explanation):
    1. semantic_search() now defensively truncates results
       to top_k and prints a WARNING if the underlying
       vector store returned more than requested. This is
       what was causing 5,000+ chunks to come back for a
       single query — the vector store's search() call was
       not honoring top_k.
    2. BM25 tokenization now strips punctuation via regex
       instead of naive .split(), so identifiers like
       "PART-001" match correctly even when followed by
       punctuation ("PART-001?", "PART-001,", "PART-001:").
"""

import os
import re
import sys
from typing import List, Dict, Optional
from collections import defaultdict

# ============================================================
# PATH SETUP
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(
    os.path.join(BASE_DIR, "..", "pipelines")
)

sys.path.append(
    os.path.join(BASE_DIR, "..", "vectorstore")
)

from embedding import embed_chunks
from vector_store import get_vector_store

from rank_bm25 import BM25Okapi


# ============================================================
# TOKENIZER (FIX #2)
# ============================================================

# Matches runs of letters/digits, so "PART-001?" -> "PART-001",
# "PART-001," -> "PART-001", "PART-001" -> "PART-001". This keeps
# alphanumeric part codes intact instead of splitting them,
# while stripping the punctuation that was breaking matches.
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


# ============================================================
# RETRIEVER
# ============================================================

class Retriever:

    def __init__(
        self,
        vector_store=None,
        backend: str = "chroma",
        embedding_backend: str = "sentence_transformers",
        embedding_model: Optional[str] = None,
        collection_name: str = "rag_documents",
    ):
        """
        Initialize the retriever.
        """

        self.store = vector_store or get_vector_store(
            backend=backend,
            collection_name=collection_name,
        )

        self.embedding_backend = embedding_backend
        self.embedding_model = embedding_model

        # BM25 data
        self.bm25 = None
        self.bm25_documents = []
        self.bm25_chunk_ids = []
        self.bm25_metadata = []

        # Build BM25 index
        self._build_bm25()

    # ========================================================
    # QUERY EMBEDDING
    # ========================================================

    def _embed_query(self, query: str) -> List[float]:
        """
        Convert the user query into an embedding using
        the SAME embedding model used for document chunks.
        """

        dummy_chunk = [
            {
                "chunk_id": "query",
                "text": query,
            }
        ]

        embedded = embed_chunks(
            dummy_chunk,
            backend=self.embedding_backend,
            model=self.embedding_model,
        )

        return embedded[0]["embedding"]

    # ========================================================
    # BUILD BM25 INDEX
    # ========================================================

    def _build_bm25(self):
        """
        Load all documents from Chroma and create a BM25 index.

        IMPORTANT:
            We preserve the original Chroma chunk_id.

        This is required for correct RRF fusion.
        """

        try:

            data = self.store.collection.get(
                include=["documents", "metadatas"]
            )

            documents = data.get("documents", [])
            ids = data.get("ids", [])
            metadatas = data.get("metadatas", [])

            if not documents:

                print("WARNING: No documents found for BM25.")

                self.bm25 = None
                return

            self.bm25_documents = documents
            self.bm25_chunk_ids = ids
            self.bm25_metadata = metadatas

            # Tokenize documents (FIX #2: regex tokenizer,
            # strips punctuation instead of naive .split())
            tokenized_documents = [
                _tokenize(document)
                for document in documents
            ]

            self.bm25 = BM25Okapi(
                tokenized_documents
            )

            print(
                f"BM25 index loaded with "
                f"{len(documents)} chunks. "
                f"(If retrieve_with_scores() ever returns "
                f"anywhere close to this number of results, "
                f"the vector store's search() is not "
                f"honoring top_k — see the WARNING in "
                f"semantic_search().)"
            )

        except Exception as e:

            print(
                "ERROR: Failed to build BM25 index."
            )

            print(e)

            self.bm25 = None

    # ========================================================
    # SEMANTIC SEARCH
    # ========================================================

    def semantic_search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Dict]:
        """
        Perform semantic vector search using Chroma.
        """

        query_embedding = self._embed_query(
            query
        )

        results = self.store.search(
            query_embedding,
            top_k=top_k,
        )

        # ----------------------------------------------------
        # FIX #1: Defensive clamp.
        #
        # This is the fix for the "5,471 chunks" bug. The
        # vector store's search() SHOULD already limit
        # results to top_k, but if it doesn't (e.g. it's
        # returning the whole collection instead of nearest
        # neighbors), this guarantees the retriever never
        # explodes downstream. The warning tells you the
        # real bug lives in vector_store.py's search().
        # ----------------------------------------------------

        if len(results) > top_k:

            print(
                f"WARNING: vector store returned "
                f"{len(results)} results but top_k={top_k} "
                f"was requested. Truncating to top_k. "
                f"This means vector_store.py's search() "
                f"method is not applying top_k correctly — "
                f"fix it there so this warning stops firing."
            )

            results = results[:top_k]

        # Add semantic rank
        for rank, result in enumerate(
            results,
            start=1
        ):
            result["semantic_rank"] = rank

            # Preserve semantic distance
            result["semantic_distance"] = result.get(
                "distance"
            )

        return results

    # ========================================================
    # BM25 SEARCH
    # ========================================================

    def bm25_search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Dict]:
        """
        Perform BM25 keyword search.

        IMPORTANT:
            BM25 results use the ORIGINAL Chroma chunk_id.
        """

        if self.bm25 is None:

            return []

        # Tokenize query (FIX #2: regex tokenizer, strips
        # punctuation so "PART-001?" matches "PART-001" in text)
        query_tokens = _tokenize(query)

        # Calculate BM25 scores
        scores = self.bm25.get_scores(
            query_tokens
        )

        # Sort highest score first
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        results = []

        for rank, index in enumerate(
            ranked_indices[:top_k],
            start=1
        ):

            chunk_id = self.bm25_chunk_ids[index]

            metadata = self.bm25_metadata[index]

            document = self.bm25_documents[index]

            results.append(
                {
                    # VERY IMPORTANT:
                    # This is the REAL Chroma chunk_id.
                    "chunk_id": chunk_id,

                    "text": document,

                    "metadata": metadata,

                    "bm25_score": float(
                        scores[index]
                    ),

                    "bm25_rank": rank,
                }
            )

        return results

    # ========================================================
    # RRF FUSION
    # ========================================================

    def rrf_fusion(
        self,
        semantic_results: List[Dict],
        bm25_results: List[Dict],
        rrf_k: int = 60,
    ) -> List[Dict]:
        """
        Combine Semantic Search and BM25 using
        Reciprocal Rank Fusion.

        Formula:

            RRF Score =
                1 / (k + semantic_rank)
                +
                1 / (k + bm25_rank)

        Only results that appear in both rankings
        receive contributions from both systems.
        """

        rrf_scores = defaultdict(float)

        combined_documents = {}

        # ----------------------------------------------------
        # SEMANTIC RESULTS
        # ----------------------------------------------------

        for result in semantic_results:

            chunk_id = result["chunk_id"]

            semantic_rank = result[
                "semantic_rank"
            ]

            rrf_scores[chunk_id] += (
                1.0
                /
                (
                    rrf_k
                    +
                    semantic_rank
                )
            )

            combined_documents[
                chunk_id
            ] = result.copy()

        # ----------------------------------------------------
        # BM25 RESULTS
        # ----------------------------------------------------

        for result in bm25_results:

            chunk_id = result["chunk_id"]

            bm25_rank = result[
                "bm25_rank"
            ]

            rrf_scores[chunk_id] += (
                1.0
                /
                (
                    rrf_k
                    +
                    bm25_rank
                )
            )

            # If BM25 found a document that
            # semantic search did not find,
            # add it to the combined collection.
            if chunk_id not in combined_documents:

                combined_documents[
                    chunk_id
                ] = result.copy()

            else:

                # Preserve BM25 information
                combined_documents[
                    chunk_id
                ]["bm25_score"] = result.get(
                    "bm25_score"
                )

                combined_documents[
                    chunk_id
                ]["bm25_rank"] = result.get(
                    "bm25_rank"
                )

        # ----------------------------------------------------
        # CREATE FINAL RESULTS
        # ----------------------------------------------------

        final_results = []

        for chunk_id, score in rrf_scores.items():

            document = combined_documents[
                chunk_id
            ]

            document["rrf_score"] = float(
                score
            )

            final_results.append(
                document
            )

        # ----------------------------------------------------
        # SORT BY RRF SCORE
        # ----------------------------------------------------

        final_results.sort(
            key=lambda x: x["rrf_score"],
            reverse=True,
        )

        # ----------------------------------------------------
        # ADD FINAL RANK
        # ----------------------------------------------------

        for rank, result in enumerate(
            final_results,
            start=1
        ):

            result["rank"] = rank

        return final_results

    # ========================================================
    # HYBRID RETRIEVAL
    # ========================================================

    def retrieve_with_scores(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 10,
    ) -> List[Dict]:
        """
        Perform:

            Semantic Search
                    +
                  BM25
                    ↓
                  RRF
                    ↓
              Final Ranking
        """

        # Semantic candidates
        semantic_results = self.semantic_search(
            query=query,
            top_k=candidate_k,
        )

        # BM25 candidates
        bm25_results = self.bm25_search(
            query=query,
            top_k=candidate_k,
        )

        # RRF
        fused_results = self.rrf_fusion(
            semantic_results=semantic_results,
            bm25_results=bm25_results,
        )

        return fused_results[:top_k]

    # ========================================================
    # RETURN CONTEXT FOR LLM
    # ========================================================

    def retrieve_as_context(
        self,
        query: str,
        top_k: int = 5,
        separator: str = "\n\n---\n\n",
    ) -> str:
        """
        Retrieve chunks and combine them into a context
        string for the LLM.
        """

        results = self.retrieve_with_scores(
            query=query,
            top_k=top_k,
        )

        return separator.join(
            result["text"]
            for result in results
        )

    # ========================================================
    # DEBUG METHOD
    # ========================================================

    def debug_retrieval(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 10,
    ):
        """
        Return all three retrieval stages separately.

        Useful for debugging:

            1. Semantic
            2. BM25
            3. RRF
        """

        semantic_results = self.semantic_search(
            query=query,
            top_k=candidate_k,
        )

        bm25_results = self.bm25_search(
            query=query,
            top_k=candidate_k,
        )

        rrf_results = self.rrf_fusion(
            semantic_results=semantic_results,
            bm25_results=bm25_results,
        )

        return {
            "semantic": semantic_results,
            "bm25": bm25_results,
            "rrf": rrf_results[:top_k],
        }


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":

    retriever = Retriever(
        backend="chroma",
        collection_name="rag_documents",
    )

    query = (
        "What are the material, machining, assembly, "
        "finishing, transportation, and profit costs "
        "specifically quoted for PART-001?"
    )

    results = retriever.retrieve_with_scores(
        query=query,
        top_k=5,
    )

    print("\n")
    print("=" * 80)
    print("QUERY")
    print("=" * 80)
    print(query)

    print("\n")
    print("=" * 80)
    print("FINAL RRF RESULTS")
    print("=" * 80)

    for result in results:

        print(
            f"\nRank: {result.get('rank')}"
        )

        print(
            f"RRF Score: "
            f"{result.get('rrf_score')}"
        )

        print(
            f"Chunk ID: "
            f"{result.get('chunk_id')}"
        )

        print(
            f"Text:\n"
            f"{result.get('text')}"
        )
