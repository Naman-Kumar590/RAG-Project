# RAG Project — Multimodal Hybrid RAG 🤖

A compact Retrieval-Augmented Generation (RAG) demo using Google Generative AI embeddings, Chroma for vector storage, and a Groq-powered chat model. This repo provides both CLI ingestion utilities and a Streamlit app that accepts PDF/DOCX/TXT uploads for on-the-fly RAG.

---

## Quick Start (2–5 minutes) ✅
1. Clone and enter the repo:

```bash
git clone <repo-url> && cd "RAG Project"
```

2. (Optional) Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
```

3. Install dependencies:

```bash
pip install -r requirements.txt
# or
pip install python-dotenv chromadb langchain_community langchain_text_splitters langchain_chroma langchain_google_genai streamlit
```

4. Add required env vars in a `.env` file at the repo root:

```env
# required
GROQ_API_KEY=your_groq_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

5. Start the Streamlit app and upload files (recommended):

```bash
streamlit run app.py
# In the sidebar: Upload PDF/DOCX/TXT files -> click "Process" -> then ask questions in the chat UI
```

Notes:
- The Streamlit app uses a **temporary per-session vector DB** by default (resets when the session/file set changes). To create a persistent DB use `ingestion_pipeline.py`.
- If you prefer a persistent DB, run:

```bash
python ingestion_pipeline.py    # builds/updates a persistent Chroma DB (e.g., dbv2/chroma_db)
```

---

## Minimal Usage
- Upload files in the Streamlit UI to process and chat with your documents (supports PDF, DOCX, TXT).
- Command-line utilities:
  - `python ingestion_pipeline.py` — ingest `Docs/` into a persistent Chroma DB
  - `python retrieval_pipeline.py` — quick single-query demo
  - `python history_aware_generation.py` — conversation/chat loop with follow-up rewriting

---

## Troubleshooting (quick tips) ⚠️
- "Missing API Keys" error: ensure `GROQ_API_KEY` and `GOOGLE_API_KEY` are present in `.env` and that `python-dotenv` is installed.
- "Error loading <file>": check required loaders (PyPDFLoader may need extra dependencies like `pypdf` or `pdfminer.six`).
- "Database resets" behavior: the Streamlit UI creates a temporary vector DB by default; upload + Process will clear and rebuild the temp DB. Use `ingestion_pipeline.py` for persistent DBs.
- Rate limits: slow down ingestion batches or retry if you get API quota errors.

---

## Key Files
- `app.py` — Streamlit UI with file upload (PDF/DOCX/TXT), per-session vector DB, chunking, BM25 index, and RAG chat UI
- `ingestion_pipeline.py` — script to ingest `Docs/` and persist a Chroma DB
- `retrieval_pipeline.py` — example single-query retrieval + generation
- `history_aware_generation.py` — chat loop with history-aware query rewriting
- `Docs/` — sample text documents used for ingestion

---

## Contributing & License
Add a `LICENSE` file to open-source this project (MIT recommended). Contributions welcome—open issues or PRs.

---

Enjoy! 🎯 — Upload documents in the sidebar, click **Process** to build the vector DB for your session, then ask questions in the chat.
