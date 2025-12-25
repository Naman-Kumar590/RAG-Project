import streamlit as st
import os
import time
import json
import base64  
from dotenv import load_dotenv

# Core Imports
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.retrievers import BM25Retriever
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.documents import Document

# --- CONFIGURATION ---
st.set_page_config(page_title="RAG Chat", page_icon="🤖")
load_dotenv()

DB_PATH = "dbv2/chroma_db"
MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"

# --- CACHED RESOURCE LOADING ---
@st.cache_resource
def initialize_system():
    # 1. Check Keys
    if not os.getenv("GROQ_API_KEY") or not os.getenv("GOOGLE_API_KEY"):
        st.error(" Missing API Keys in .env file!")
        st.stop()
        
    # 2. Load Vector DB
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    if not os.path.exists(DB_PATH):
        st.error(f" Database not found at {DB_PATH}")
        st.stop()
        
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    
    # 3. Build BM25 Index
    try:
        existing_data = db.get()
        docs_for_bm25 = [
            Document(page_content=txt, metadata=meta)
            for txt, meta in zip(existing_data['documents'], existing_data['metadatas'])
        ]
        bm25_retriever = BM25Retriever.from_documents(docs_for_bm25)
        bm25_retriever.k = 5
    except Exception as e:
        st.error(f" BM25 Build Failed: {e}")
        st.stop()
        
    # 4. Initialize Chat Model
    model = ChatGroq(model=MODEL_NAME, temperature=0)
    
    return db, bm25_retriever, model

db, bm25_retriever, model = initialize_system()

# --- HELPER FUNCTIONS ---

def reciprocal_rank_fusion(vector_docs, bm25_docs, k=60):
    fused_scores = {}
    doc_map = {}
    
    def update_score(docs, weight=1.0):
        for rank, doc in enumerate(docs):
            doc_key = doc.page_content[:200]
            if doc_key not in doc_map: doc_map[doc_key] = doc
            score = 1 / (rank + k)
            fused_scores[doc_key] = fused_scores.get(doc_key, 0) + (score * weight)

    update_score(vector_docs)
    update_score(bm25_docs)
    
    sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[key] for key, score in sorted_docs[:3]]

def safe_api_call(func, *args, **kwargs):
    for i in range(3):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if "429" in str(e): time.sleep(1)
            else: raise e
    return None

def rewrite_query(history, question):
    if not history: return question
    msg = [SystemMessage(content="Rewrite as standalone search query."), *history, HumanMessage(content=question)]
    res = safe_api_call(model.invoke, msg)
    return res.content.strip() if res else question

def generate_answer(docs, history, question):
    context = ""
    for i, doc in enumerate(docs):
        try:
            meta = json.loads(doc.metadata.get("original_content", "{}"))
        except:
            meta = {}
        context += f"\n[Doc {i+1}]: {meta.get('raw_text', doc.page_content)}\n"
        for t in meta.get("tables_html", []): context += f"\n{t}\n"

    prompt = f"Answer using this context:\n{context}\n\nQuestion: {question}"
    msg = [SystemMessage(content="You are a helpful assistant."), *history, HumanMessage(content=prompt)]
    res = safe_api_call(model.invoke, msg)
    return res.content.strip() if res else "Error generating answer."

# --- UI LAYOUT ---

st.title(" Multimodal Hybrid RAG")
st.caption(f"Powered by Groq ({MODEL_NAME}) + Google Embeddings + BM25")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about documents (try 'Show me Figure 1')..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Prepare History
        history_lc = []
        for msg in st.session_state.messages[:-1]:
            if msg["role"] == "user": history_lc.append(HumanMessage(content=msg["content"]))
            else: history_lc.append(AIMessage(content=msg["content"]))

        # --- PROCESSING ---
        with st.status(" Processing...", expanded=True) as status:
            st.write(" Optimizing query...")
            search_query = rewrite_query(history_lc[-2:], prompt)
            
            st.write(" Running Hybrid Search...")
            vector_docs = db.similarity_search(search_query, k=5)
            bm25_docs = bm25_retriever.invoke(search_query)
            final_docs = reciprocal_rank_fusion(vector_docs, bm25_docs)
            
            st.write(f" Found {len(final_docs)} relevant chunks.")
            status.update(label=" Answer Ready!", state="complete", expanded=False)

        # --- GENERATE TEXT ANSWER ---
        full_response = generate_answer(final_docs, history_lc[-2:], prompt)
        message_placeholder.markdown(full_response)
        
        # --- SMART VISUALS LOGIC ---
        # 1. Detect Intent: Did the user ask for visual content?
        visual_keywords = ["image", "figure", "graph", "chart", "table", "diagram", "show me"]
        user_wants_visuals = any(kw in prompt.lower() for kw in visual_keywords)
        
        # 2. Render (Auto-expand if asked, otherwise collapsed)
        if final_docs:
            with st.expander(" Retrieved Images & Tables", expanded=user_wants_visuals):
                for i, doc in enumerate(final_docs):
                    try:
                        meta = json.loads(doc.metadata.get("original_content", "{}"))
                        
                        # Tables
                        for table_html in meta.get("tables_html", []):
                            st.caption(f"**Table from Document {i+1}**")
                            st.markdown(table_html, unsafe_allow_html=True)
                            st.divider()
                        
                        # Images
                        for img_b64 in meta.get("images_base64", []):
                            st.caption(f"**Figure from Document {i+1}**")
                            st.image(
                                base64.b64decode(img_b64), 
                                use_column_width=True
                            )
                            st.divider()
                    except Exception as e:
                        st.warning(f"Error rendering media: {e}")

    st.session_state.messages.append({"role": "assistant", "content": full_response})