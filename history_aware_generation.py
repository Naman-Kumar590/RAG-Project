from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

persistant_directory = "db/chroma_db"
embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
db = Chroma(
    persist_directory=persistant_directory,
    embedding_function=embedding_model,
)

model = ChatGoogleGenerativeAI(model="gemini-flash-latest")

chat_history = []

def get_clean_text(response_content):
    # Case 1: It's a list (like the output you just got)
    if isinstance(response_content, list):
        text_parts = []
        for part in response_content:
            # If the item is a dictionary (e.g. {'type': 'text', 'text': '...'})
            if isinstance(part, dict):
                text_parts.append(part.get('text', ''))
            # If the item is just a string inside a list
            else:
                text_parts.append(str(part))
        return "".join(text_parts).strip()
    
    # Case 2: It's already a string
    return str(response_content).strip()

def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")
    
    # Step 1: Make the question clear using conversation history
    if chat_history:
        # Ask AI to make the question standalone
        messages = [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."),
        ] + chat_history + [
            HumanMessage(content=f"New question: {user_question}")
        ]
        
        result = model.invoke(messages)
        if isinstance(result.content, list):
            # If Gemini returns a list of parts, join them into one string
            search_question = " ".join([str(item) for item in result.content]).strip()
        else:
        # If it returns a normal string, just use it
            search_question = result.content.strip()
            print(f"Searching for: {search_question}")
    else:
        search_question = user_question
    
    # Step 2: Find relevant documents
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(search_question)
    
    print(f"Found {len(docs)} relevant documents:")
    for i, doc in enumerate(docs, 1):
        # Show first 2 lines of each document
        lines = doc.page_content.split('\n')[:2]
        preview = '\n'.join(lines)
        print(f"  Doc {i}: {preview}...")
    
    # Step 3: Create final prompt
    combined_input = f"""Based on the following documents, please answer this question: {user_question}

    Documents:
    {"\n".join([f"- {doc.page_content}" for doc in docs])}

    Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
    """
    
    # Step 4: Get the answer
    messages = [
        SystemMessage(content="You are a helpful assistant that answers questions based on provided documents and conversation history."),
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]
    
    result = model.invoke(messages)
    answer = get_clean_text(result.content)
    
    # Step 5: Remember this conversation
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))
    
    print(f"Answer: {answer}")
    return answer


def start_chat():
    print("Ask me Questions ! Type 'exit' to quit.")

    while True:
        question = input("You: ")
        if question.lower() == 'exit':
            print("Exiting chat. Goodbye!")
            break

        answer = ask_question(question)

if __name__ == "__main__":
    start_chat()
