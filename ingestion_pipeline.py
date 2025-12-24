import os
import time
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv


load_dotenv()

def load_documents(docs_path="Docs"):

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The specified path does not exist: {docs_path}")
    
    loader = DirectoryLoader(
        path = docs_path,
        glob = "*.txt",
        loader_cls = TextLoader,
        loader_kwargs={'encoding': 'utf-8'}
    )

    documents = loader.load()
    if len(documents) == 0:
        raise ValueError(f"No documents found in the specified path: {docs_path}")
    
    # for i, doc in enumerate(documents[:2]):
    #     print(f"Document {i+1}:")
    #     print(f"Source: {doc.metadata['source']}")
    #     print(f"Content Length: {len(doc.page_content)} characters\n")
    #     print(f"Content Preview:\n{doc.page_content[:100]}...\n")
    #     print(f"metadata: {doc.metadata}\n")
        
    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    print("Splitting Documents into chunks...")
    text_splitter = CharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)

    # if chunks:
    #     for i, chunk in enumerate(chunks[:5]):
    #         print(f"\n--- Chunk {i+1} ---")
    #         print(f"Source: {chunk.metadata['source']}")
    #         print(f"Chunk Length: {len(chunk.page_content)} characters")
    #         print(f"Content:")
    #         print(chunk.page_content)
    #         print("-"* 50)
        
    #     if(len(chunks) > 5):
    #         print(f"\n...and {len(chunks) - 5} more chunks.")
    
    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    print("Creating Vector Store...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    # 1. Initialize an EMPTY Vector Store
    # We remove 'documents=chunks' so it doesn't upload anything yet.
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    # 2. Use the Batched Loop to fill it safely
    batch_size = 15
    total_chunks = len(chunks)

    print(f"Processing {total_chunks} chunks in batches of {batch_size}...")

    for i in range(0, total_chunks, batch_size):
        # Slice the list to get just 15 items
        batch = chunks[i : i + batch_size]
        
        # Add the batch to the database
        vectorstore.add_documents(batch)
        
        print(f"  - Added batch {i // batch_size + 1} ({len(batch)} chunks)")
        
        # WAIT! Pause to respect the API limit
        time.sleep(3)

    print("---Finished creating Vector Store---")
    print(f"Vector store created and persisted at: {persist_directory}")
    return vectorstore

def main():
    documents = load_documents(docs_path="Docs")
    chunks = split_documents(documents)
    vectorstore = create_vector_store(chunks)

if __name__ == "__main__":
    main() 

