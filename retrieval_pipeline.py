from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage , SystemMessage

load_dotenv()

persistant_directory = "db/chroma_db"

embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

db = Chroma(
    persist_directory=persistant_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

query = "what was microsoft's first hardware product release?"

retriever = db.as_retriever(search_type = "mmr",search_kwargs={"k":5})

relevant_docs = retriever.invoke(query)

print(f"user query: {query}\n")

print("---context retrieved---\n")

for i, doc in enumerate(relevant_docs):
    print(f"Document {i}:\n{doc.page_content}\n")

combined_input = f"""based on the following Document answer the user query: {query}

Documents:
{chr(10).join([f"-{doc.page_content}" for doc in relevant_docs])}

Please provide a clear and concise answer from these documents.If you cannot find the answer in the documents, please respond with "I don't know"."""

model = ChatGoogleGenerativeAI(model="gemini-flash-latest")

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input)
]

# Invoke the model with the messages
result = model.invoke(messages)

# Print the result
print("\n---Generated Answer---\n")
print("content only:")
print(result.content)