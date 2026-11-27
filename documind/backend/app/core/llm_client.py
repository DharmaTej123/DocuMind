"""
LLM client abstraction.

Agent nodes call `get_llm()` and use the returned object — they never import
the OpenAI SDK (or any provider SDK) directly. Swapping providers later
means editing this file only, per docs/REQUIREMENTS.md Decisions Log.

Providers are looked up in _BUILDERS, keyed by the LLMProvider enum. The check
at the bottom fails at import time if an enum member has no builder, so a new
provider can't be added to the config and forgotten here.
"""
from app.core.config import LLMProvider, get_settings


def _gemini(settings):
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        temperature=0,
    )


def _anthropic(settings):
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=settings.anthropic_model,
        api_key=settings.anthropic_api_key,
        temperature=0,
        max_tokens=1024,
    )


def _openai(settings):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        base_url="https://openai.vocareum.com/v1",
        temperature=0,
    )


def _groq(settings):
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=0,
    )


_BUILDERS = {
    LLMProvider.GEMINI: _gemini,
    LLMProvider.ANTHROPIC: _anthropic,
    LLMProvider.OPENAI: _openai,
    LLMProvider.GROQ: _groq,
}

_missing = set(LLMProvider) - set(_BUILDERS)
if _missing:
    raise RuntimeError(f"llm_client has no builder for provider(s): {sorted(p.value for p in _missing)}")


def get_llm():
    """
    Returns a configured chat model instance based on settings.llm_provider.
    Uses LangChain's chat model interface so it drops directly into LangGraph nodes.
    """
    settings = get_settings()
    return _BUILDERS[settings.llm_provider](settings)
