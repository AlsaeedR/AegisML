import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    """
    Initialize the LLM used by the Reporting Agent.
    """

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. "
            "Please ensure it is defined in your "
            "environment or .env file."
        )

    model_name = os.getenv(
        "OPENAI_MODEL_NAME",
        "gpt-5.4-mini",
    )

    return ChatOpenAI(
        api_key=api_key,
        model=model_name,
        temperature=temperature,
    )