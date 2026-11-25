"""
LLM client abstraction.

Agent nodes call `get_llm()` and use the returned object — they never import
the OpenAI SDK (or any provider SDK) directly. Swapping providers later
(e.g. adding Claude) means editing this file only, per docs/REQUIREMENTS.md
Decisions Log.
"""
from app.core.config import get_settings


def get_llm():
    """
    Returns a configured chat model instance based on settings.llm_provider.
    Uses LangChain's chat model interface so it drops directly into LangGraph nodes.
    """
    settings = get_settings()

    if settings.llm_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=0,
        )

    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key,
            temperature=0,
            max_tokens=1024,
        )

    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url="https://openai.vocareum.com/v1",
            temperature=0,
        )
    if settings.llm_provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=0,
        )

    raise ValueError(f"Unsupported llm_provider: {settings.llm_provider}")
