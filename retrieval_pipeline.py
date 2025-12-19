from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

persistant_directory = "db/chroma_db"

embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

db = Chroma(
    persist_directory=persistant_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

query = "in what year did tesla begin production of the roadster?"

retriever = db.as_retriever(search_kwargs={"k":5})

relevant_docs = retriever.invoke(query)

print(f"user query: {query}\n")

print("---context retrieved---\n")

for i, doc in enumerate(relevant_docs):
    print(f"Document {i}:\n{doc.page_content}\n")

