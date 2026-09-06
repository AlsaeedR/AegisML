import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def get_llm():
    """
    LangChain LLM wrapper for Agent 1.
    """
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set in environment.")

    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model="gpt-4.1-mini",
        temperature=0.2,
    )
    return llm
