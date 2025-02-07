import sys
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from .config import OPENAI_API_KEY

def knowledge_chat_session():
    """
    Launch an advanced interactive chat session with context retention.
    - Type 'clear' to reset the conversation history.
    - Type 'exit' or 'quit' to end chat mode.
    Designed for scientific Q&A.
    """
    chat = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        temperature=0.7
    )

    messages = [
        SystemMessage(content=(
            "You are an advanced scientific AI assistant. Answer questions thoroughly, accurately, and concisely."
        ))
    ]

    print("\n=== Advanced Chat Mode ===")
    print("Enter your question below. Type 'clear' to reset conversation, or 'exit'/'quit' to end chat.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chat mode.\n")
            break

        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chat mode.\n")
            break
        if user_input.lower() == "clear":
            messages = [
                SystemMessage(content=(
                    "You are an advanced scientific AI assistant. Answer questions thoroughly, accurately, and concisely."
                ))
            ]
            print("Conversation history cleared.\n")
            continue
        if not user_input:
            continue

        messages.append(HumanMessage(content=user_input))
        try:
            response = chat.invoke(messages)
        except Exception as e:
            print("Error during chat invocation:", e)
            continue

        print("osairo:", response.content, "\n")
        messages.append(AIMessage(content=response.content))
