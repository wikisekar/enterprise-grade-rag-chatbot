"""
generator.py

The missing piece: takes a user's question, retrieves relevant chunks
using the existing Retriever, and asks an LLM to write an actual answer
grounded ONLY in those chunks.

This version uses Groq's free API instead of a local model. Only the
question + a few small retrieved text snippets are sent to Groq's
servers — your documents, project files, and vector database all stay
on your machine. This exists purely because local inference (via
Ollama) was too slow on this machine's older 2-core CPU.

Setup (one-time, free, no credit card):
  1. Sign up at https://console.groq.com and create an API key
  2. Add to your .env file (project root):
       GROQ_API_KEY=your_key_here
  3. pip install groq python-dotenv
"""

import os
import sys

from groq import Groq
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), "retrieval"))
from retriever import Retriever  # noqa: E402


load_dotenv()  # reads GROQ_API_KEY from your .env file

# Free-tier model on Groq, fast and solid quality for RAG synthesis.
MODEL = "llama-3.3-70b-versatile"

# Matches the threshold tuned in retriever.py / eval_retrieval.py
DISTANCE_THRESHOLD = 0.7

SYSTEM_PROMPT = """You are a document Q&A assistant for a business's internal \
document collection (contracts, tax filings, invoices, correspondence, etc).

Rules you MUST follow:
1. Answer ONLY using the provided context below. Do not use outside knowledge.
2. If the context does not contain enough information to answer the question, \
say so clearly instead of guessing.
3. Be concise and direct. Quote specific figures, dates, or clause numbers \
from the context when relevant.
4. Do not mention "the context" or "the provided chunks" in your answer — \
just answer naturally, as if you already knew this from the documents."""


class Generator:
    """
    Wraps Retriever + Groq API: question in, grounded answer out.
    """

    def __init__(self, model: str = MODEL,
                 distance_threshold: float = DISTANCE_THRESHOLD):

        self.retriever = Retriever(backend="chroma", collection_name="rag_documents")
        self.client = Groq()  # reads GROQ_API_KEY from env automatically
        self.model = model
        self.distance_threshold = distance_threshold

    def answer(self, question: str, top_k: int = 10) -> str:
        """
        Retrieve relevant chunks for the question and ask Groq to
        answer using only that context.
        """

        context = self.retriever.retrieve_as_context(
            question,
            top_k=top_k,
        )

        if not context.strip():
            return ("I couldn't find anything relevant to that question in "
                    "the document collection.")

        user_message = f"""Context from the document collection:

{context}

---

Question: {question}"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )

        return response.choices[0].message.content


if __name__ == "__main__":

    generator = Generator()

    print("RAG chatbot ready. Type a question (or 'quit' to exit).\n")

    while True:

        question = input("You: ").strip()

        if question.lower() in ("quit", "exit"):
            break

        if not question:
            continue

        print("\nThinking...\n")
        answer = generator.answer(question)
        print(f"Assistant: {answer}\n")