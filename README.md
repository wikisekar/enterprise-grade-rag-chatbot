# Advanced RAG Enterprise Chatbot

An enterprise-focused Retrieval-Augmented Generation (RAG) system designed to ingest documents from multiple sources, process and clean the content, generate embeddings, store them in a vector database, and retrieve the most relevant information for downstream LLM-based question answering.

---

## 🚀 Project Overview

This project implements an end-to-end RAG pipeline for enterprise document retrieval.

The system is designed to ingest documents from various sources (e.g. email, file uploads), including PDFs, Word documents, and other supported file formats.

Instead of sending an entire document collection directly to an LLM, the system:

1. Ingests documents from various sources
2. Extracts and processes document content
3. Cleans the extracted text
4. Splits documents into meaningful chunks
5. Generates vector embeddings
6. Stores the embeddings in a vector database
7. Performs semantic retrieval
8. Retrieves the most relevant chunks for a future LLM response

The current implementation has been completed through the **retrieval stage**.

---

# 🏗️ Architecture

```text
                Document Source
                      │
                      ▼
             Document Ingestion
                      │
                      ▼
          Document Processing
                      │
                      ▼
             Text Cleaning
                      │
                      ▼
              Chunking
                      │
                      ▼
             Embeddings
                      │
                      ▼
             Vector Database
                      │
                      ▼
          Semantic Retrieval
                      │
                      ▼
                  Relevant
                   Chunks
                      │
                      ▼
                    LLM
               (Next Stage)
                      │
                      ▼
             Generated Answer

---

## Additional Capabilities (Production Deployment)

Beyond the pipeline stages shown in this repository, the full production system (deployed on AWS) also includes:

- **Multi-channel document ingestion** � supports bulk upload of hundreds to thousands of documents (PDF, Word, etc.) directly to cloud storage, in addition to automated email-based ingestion
- **WhatsApp Business integration** � enables conversational document submission and query handling via WhatsApp
- **Automated quotation generation** � generates business quotations dynamically based on retrieved context and client requirements
- **Cloud-native pipeline orchestration** � the full ingestion-to-retrieval pipeline runs as an automated, scheduled pipeline on AWS

*Note: The ingestion, storage, and business-logic modules for these features are part of a private production deployment and are not included in this repository to protect client confidentiality. This repo showcases the core, reusable RAG pipeline architecture (cleaning, chunking, embedding, retrieval).*
