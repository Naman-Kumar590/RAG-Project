# Multi-Modal RAG 📚🔬

## Overview
This document explains the multimodal Retrieval-Augmented Generation (RAG) pipeline implemented in this repository (see `multi_model_rag.ipynb`). The pipeline supports text, tables, and images extracted from PDFs and uses Google Generative AI embeddings + Chroma for retrieval and Gemini models for generation.

---

## Key components
- Partition: `partition_pdf` (Unstructured) to extract text, images, and tables from PDFs.
- Chunking: `chunk_by_title` to create intelligent, context-aware chunks.
- Summarisation: `summarise_chunks` which includes robust fallbacks and safety helpers (`clean_content_to_string`).
- Vector Store: `Chroma` with `GoogleGenerativeAIEmbeddings` to persist embeddings.
- Generation: `ChatGoogleGenerativeAI` (Gemini) to produce answers that reference text, tables, and images.

---

## How it works (high level)
1. Partition the PDF into parsed elements (text blocks, tables, images).
2. Chunk the document using title-aware logic so chunks are coherent and no larger than ~3000 characters.
3. For chunks with images/tables, attempt to create an AI-enhanced summary; otherwise use raw text.
4. Convert chunk content to a safe string and construct `langchain_core` `Document` objects.
5. Create embeddings and persist them with Chroma.
6. For queries: retrieve top-k chunks and send both text and images to Gemini (as text + image payloads) to produce a multimodal answer.

---

## Notebook mapping (`multi_model_rag.ipynb`)
- `partition_document` — example of using `partition_pdf` with `strategy='hi_res'` to preserve structure.
- `create_chunks_by_title` — uses `chunk_by_title` to produce chunks.
- `summarise_chunks` — a safe summariser with `clean_content_to_string` to avoid non-string inputs.
- `create_vector_store` — creates a Chroma DB in `dbv1/` or `dbv2/`.
- `generate_final_answer` — composes a text prompt including TABLE HTML and image base64 payloads and invokes `ChatGoogleGenerativeAI`.

---

## Usage
- From the notebook: run the cells in order (partition → chunk → summarise → create store → query → generate).
- Programmatically: call `run_complete_ingestion_pipeline(pdf_path)` and then use the returned `db` to run retrieval and `generate_final_answer`.

Example:

```python
from multi_model_rag import run_complete_ingestion_pipeline
# Build DB
db = run_complete_ingestion_pipeline('./Docs/your-pdf.pdf')
# Query
retriever = db.as_retriever(search_kwargs={"k": 3})
chunks = retriever.invoke("What are the main components of the Transformer?")
# Generate an answer (notebook shows a helper)
```

---

## Dependencies
- unstructured (for `partition_pdf`)
- langchain_core, langchain_chroma, langchain_google_genai
- chromadb
- python-dotenv

Make sure your `.env` contains the necessary Google credentials.

---

## Troubleshooting & tips
- ValidationError (input should be string): Use `clean_content_to_string` to guarantee strings for LLM calls.
- Images not showing in generation: ensure image payloads are base64-encoded and included as `image_url` entries when building the chat message.
- Large PDFs: use `max_characters` and `new_after_n_chars` to tune chunk sizes.

---

## Short checklist ✅
- [ ] PDF partitions correctly (text, tables, images)
- [ ] Chunks created and cleaned
- [ ] Vector store persisted (check `dbv1/` or `dbv2/`)
- [ ] Multimodal generation returns expected results

---

© RAG Project
