# RAG Project

✅ **Repository**: A small Retrieval-Augmented Generation (RAG) demo using Google Generative AI embeddings and Chroma for vector storage.

---

## 🔧 Overview
This project demonstrates a simple RAG pipeline: ingest text files into a Chroma vector store, retrieve relevant chunks for a query, and generate answers using a Google Generative AI chat model (Gemini family).

---

## 📁 Project Structure
- `ingestion_pipeline.py` — Load `.txt` files from `Docs/`, split into chunks, and create/persist a Chroma vector store in `db/chroma_db`.
- `retrieval_pipeline.py` — Example script to query the vector store and generate an answer with the chat model.
- `history_aware_generation.py` — A simple chat loop that maintains conversation history, rewrites follow-ups to standalone questions, retrieves relevant documents, and returns an answer.
- `Docs/` — Source text documents used for ingestion (e.g., `Google.txt`, `Microsoft.txt`, ...).
- `db/chroma_db/` — Persisted Chroma database files produced by the ingestion step.

---

## ⚙️ Prerequisites
- Python 3.8+ (recommended)
- Create a virtual environment (optional but recommended):

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

- Install required packages (example):

```bash
pip install python-dotenv langchain_chroma langchain_google_genai langchain_core langchain_community langchain_text_splitters chromadb
```

> Tip: If the project includes a `requirements.txt`, prefer `pip install -r requirements.txt`.

---

## 🔑 Setup
1. Create a `.env` file in the project root with your Google Generative AI credentials (the exact variables depend on your Google credentials setup). Example:

```
# .env (example)
GOOGLE_API_KEY=your_api_key_here
```

2. Make sure `Docs/` contains the text files you want to index (the repository already has sample files).

---

## ▶️ Usage
1. Ingest documents and build the vector store:

```bash
python ingestion_pipeline.py
```

2. Run a single-query retrieval + generation demo:

```bash
python retrieval_pipeline.py
```

3. Start the interactive chat with history-aware question rewriting:

```bash
python history_aware_generation.py
```

---

## 📌 Implementation Notes
- Embeddings: `GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")` (see code in `ingestion_pipeline.py`).
- Chat model: `ChatGoogleGenerativeAI(model="gemini-flash-latest")` (used in `retrieval_pipeline.py` and `history_aware_generation.py`).
- The ingestion pipeline batches document uploads (default batch size 15) and pauses between batches to respect API limits.
- `ingestion_pipeline.py` raises helpful errors if `Docs/` is missing or empty.
- `history_aware_generation.py` tries to rewrite follow-up questions to be standalone for better retrieval.

---

## 🛠 Troubleshooting
- "No documents found": confirm text files exist in `Docs/` and are UTF-8 encoded.
- Authentication errors: confirm `.env` contains valid Google Generative AI credentials and that required environment variables are loaded.
- If embeddings or model names change, update the strings used in the scripts.

---

## ✅ Next steps / Ideas
- Add a `requirements.txt` and CI checks.
- Add unit/integration tests and example queries for reproducibility.
- Add a small `Makefile` or task runner to automate ingestion and tests.

---

If you'd like, I can also:
- Add a `requirements.txt` with pinned versions.
- Commit the README to git and create a branch + PR.

---

© RAG Project
