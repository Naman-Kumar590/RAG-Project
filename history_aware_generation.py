import os
import time
import json
from typing import List, Dict
from dotenv import load_dotenv

# Core Imports
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.retrievers import BM25Retriever
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.documents import Document

# --- CONFIGURATION ---
load_dotenv()

DB_PATH = "dbv2/chroma_db"
# Use the Llama 4 Scout model (or your preferred Groq model)
MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"

print("🔧 Initializing Manual Hybrid RAG (RRF)...")

# 1. SETUP VECTOR DB
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
if os.path.exists(DB_PATH):
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    print(f"✅ Loaded Vector DB from: {DB_PATH}")
else:
    print(f"❌ CRITICAL ERROR: Database not found at {DB_PATH}")
    exit(1)

# 2. SETUP BM25 INDEX
print("   🔨 Building BM25 Index...")
try:
    existing_data = db.get()
    docs_for_bm25 = [
        Document(page_content=txt, metadata=meta)
        for txt, meta in zip(existing_data['documents'], existing_data['metadatas'])
    ]
    if not docs_for_bm25:
        raise ValueError("Database is empty")
    
    bm25_retriever = BM25Retriever.from_documents(docs_for_bm25)
    bm25_retriever.k = 5 # Get top 5 keywords
    print(f"   ✅ Indexed {len(docs_for_bm25)} documents for BM25.")
except Exception as e:
    print(f"❌ BM25 Build Failed: {e}")
    exit(1)

# 3. SETUP CHAT MODEL
model = ChatGroq(model=MODEL_NAME, temperature=0)

# --- HYBRID SEARCH LOGIC (RRF) ---

def reciprocal_rank_fusion(vector_docs: List[Document], bm25_docs: List[Document], k=60):
    """
    Combines results from two sources using Reciprocal Rank Fusion.
    """
    fused_scores = {}
    doc_map = {}

    # Helper to update scores
    def update_score(docs, weight=1.0):
        for rank, doc in enumerate(docs):
            # Use content as a unique key (simple approach)
            doc_key = doc.page_content[:200]  # First 200 chars as key
            if doc_key not in doc_map:
                doc_map[doc_key] = doc
            
            # RRF Score Formula: 1 / (rank + k)
            score = 1 / (rank + k)
            fused_scores[doc_key] = fused_scores.get(doc_key, 0) + (score * weight)

    # 1. Process Vector Results
    update_score(vector_docs)
    
    # 2. Process BM25 Results
    update_score(bm25_docs)

    # 3. Sort by combined score
    sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return top 3 unique docs
    return [doc_map[key] for key, score in sorted_docs[:3]]

def hybrid_search(query: str):
    """Runs both searches and fuses them."""
    # 1. Vector Search
    vector_docs = db.similarity_search(query, k=5)
    
    # 2. Keyword Search
    bm25_docs = bm25_retriever.invoke(query)
    
    # 3. Manual Fusion
    return reciprocal_rank_fusion(vector_docs, bm25_docs)

# --- CHAT LOGIC ---

def safe_api_call(func, *args, **kwargs):
    for i in range(3):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if "429" in str(e):
                time.sleep(2 * (i + 1))
            else:
                raise e
    raise Exception("API Failed")

def rewrite_query(history, question):
    if not history: return question
    msg = [SystemMessage(content="Rewrite as standalone query."), *history, HumanMessage(content=question)]
    return safe_api_call(model.invoke, msg).content.strip()

def generate_answer(docs, history, question):
    context = ""
    for i, doc in enumerate(docs):
        # Handle metadata safely
        try:
            meta = json.loads(doc.metadata.get("original_content", "{}"))
        except:
            meta = {}
        
        context += f"\n[Doc {i+1}]: {meta.get('raw_text', doc.page_content)}\n"
        for t in meta.get("tables_html", []): context += f"\n{t}\n"

    prompt = f"Answer using this context:\n{context}\n\nQuestion: {question}"
    msg = [SystemMessage(content="Helpful Assistant"), *history, HumanMessage(content=prompt)]
    return safe_api_call(model.invoke, msg).content.strip()

def start_chat():
    history = []
    print("\n🚀 MANUAL HYBRID SYSTEM READY!")
    print("   Method: Reciprocal Rank Fusion (Vector + BM25)\n")

    while True:
        q = input("You: ")
        if q.lower() in ["exit", "quit"]: break
        
        try:
            # 1. Rewrite
            q_rewritten = rewrite_query(history[-2:], q)
            print(f"   🔍 Searching: {q_rewritten}")
            
            # 2. Hybrid Retrieval (The Manual Part)
            best_docs = hybrid_search(q_rewritten)
            
            # 3. Generate
            print("   🤖 Thinking...")
            ans = generate_answer(best_docs, history[-2:], q)
            print(f"\nRAG: {ans}\n")
            
            history.extend([HumanMessage(content=q), AIMessage(content=ans)])
            
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    start_chat()