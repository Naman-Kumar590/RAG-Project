# RAG Project — Multimodal Hybrid RAG 🤖

A compact Retrieval-Augmented Generation (RAG) demo using Google Generative AI embeddings, Chroma for vector storage, and a Groq-powered chat model. This repo includes an ingestion pipeline, retrieval utilities, and a Streamlit app for interactive querying of ingested documents.

---

## Quick Start (3–5 minutes) ✅
1. Clone and enter the repo:

```bash
git clone <repo-url> && cd "RAG Project"
```

2. (Optional) Create a Python virtual environment and activate it:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
```

3. Install dependencies:

```bash
pip install -r requirements.txt
# or
pip install python-dotenv chromadb langchain_chroma langchain_google_genai streamlit
```

4. Add required env vars in a `.env` file at the repo root:

```env
# required
GROQ_API_KEY=your_groq_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

5. Build the vector DB (ingest documents) and run the app:

```bash
python ingestion_pipeline.py       # creates/updates a Chroma DB (dbv2/chroma_db)
streamlit run app.py               # open the interactive RAG UI at http://localhost:8501
```

---

## Minimal Usage
- Ask document-aware questions in the Streamlit UI (e.g., "Show me Figure 1", "Summarize the Tesla doc").
- Command-line utilities:
  - `python retrieval_pipeline.py` — quick single-query demo
  - `python history_aware_generation.py` — conversation/chat loop with follow-up rewriting

---

## Troubleshooting (quick tips) ⚠️
- "Missing API Keys" error: ensure `GROQ_API_KEY` and `GOOGLE_API_KEY` are in `.env` and loaded.
- "Database not found" error: run `python ingestion_pipeline.py` to recreate `dbv2/chroma_db`.
- Rate limits: ingestion respects pauses; retry or increase batch delays if you see API limit errors.

---

## Key Files
- `app.py` — Streamlit UI (uses `dbv2/chroma_db` by default)
- `ingestion_pipeline.py` — ingest docs from `Docs/` into Chroma
- `retrieval_pipeline.py` — example single-query retrieval/generation
- `history_aware_generation.py` — chat loop with history-aware query rewriting
- `Docs/` — sample text documents (used for ingestion)

---

## License & Contributing
Add a `LICENSE` file if you want to open-source this project. Contributions and issues are welcome — open a PR or issue describing the change.

---

Enjoy! 🎯 — Run the ingestion step to build your DB, then ask questions via the Streamlit app or the example scripts.
