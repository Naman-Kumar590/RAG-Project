# History-Aware Generation 🤖💬

## Overview
This document explains `history_aware_generation.py`, a small chat loop that maintains conversation history, rewrites follow-up questions into standalone queries for better retrieval, and returns answers based only on retrieved documents.

---

## Key components
- `ChatGoogleGenerativeAI` (Gemini) — used both for rewriting follow-ups and final answer generation.
- `Chroma` — vector store used as the retriever backend.
- `chat_history` — a simple in-memory history of the conversation.
- `get_clean_text` — helper to extract text when model responses come as lists/dicts.

---

## How it works
1. User asks a question (could be follow-up).
2. If `chat_history` is non-empty, the assistant uses a short system prompt to ask Gemini to rewrite the new question to be standalone and searchable.
3. The rewritten question is used to query Chroma (`retriever.invoke(search_question)`), returning top-k documents.
4. The final prompt is composed from the retrieved documents and fed to Gemini to produce the answer.
5. Both the user question and the assistant answer are appended to `chat_history` to be used for later rewrites.

---

## Practical usage
- Run the script:

```bash
python history_aware_generation.py
```

- Type questions interactively. Type `exit` to quit.

Customization:
- Change `persistant_directory` to point to a different Chroma persistence folder.
- Adjust embedding model/LLM by replacing `GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")` or the `ChatGoogleGenerativeAI` model selection.
- Change retriever `k` by editing `retriever = db.as_retriever(search_kwargs={"k": N})`.

---

## Common pitfalls
- Model returns list/dict pieces: `get_clean_text` merges list parts into a string.
- No documents found: ensure a vector store exists in the selected directory and has been populated (run `ingestion_pipeline.py`).

---

## Example
- Ask: "Who developed the Transformer model?"
- Follow-up: "How many heads does it have?" — the script will rewrite the second question to be standalone and then perform retrieval.

---

© RAG Project
