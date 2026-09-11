import os
from typing import Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def is_llm_available() -> bool:
    """
    Check if LLM credentials are configured in the environment.
    """
    return bool(os.getenv("OPENAI_API_KEY"))


def get_llm(
    temperature: float = 0.2,
    model_name: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> ChatOpenAI:
    """
    Centralized factory for initializing LangChain ChatOpenAI instances across all agents.
    
    Args:
        temperature: Sampling temperature (e.g. 0.0 for deterministic planning, 0.2 for narrative).
        model_name: Optional model override. Defaults to OPENAI_MODEL_NAME or 'gpt-5.4-mini'.
        max_tokens: Optional token generation limit.
        
    Returns:
        Configured ChatOpenAI client.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. "
            "Please ensure it is defined in your environment or .env file."
        )

    resolved_model = model_name or os.getenv("OPENAI_MODEL_NAME", "gpt-5.4-mini")

    kwargs = {
        "api_key": api_key,
        "model": resolved_model,
        "temperature": temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    return ChatOpenAI(**kwargs)

