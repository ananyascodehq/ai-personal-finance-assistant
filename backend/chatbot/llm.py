"""Pick and start the chat model.

Settings come from .env:
    LLM_PROVIDER     groq | ollama          which model answers first
    LLM_FALLBACK     ollama | none          backup if the main model fails
    GROQ_API_KEY     needed when Groq is used
    GROQ_MODEL       e.g. openai/gpt-oss-120b
    OLLAMA_MODEL     e.g. qwen3:8b
    OLLAMA_BASE_URL  default http://localhost:11434
"""

import logging
import os

import ollama
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

load_dotenv()

log = logging.getLogger(__name__)

PROVIDERS = ("groq", "ollama")
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"


class LLMConfigError(RuntimeError):
    """Raised when the chosen provider can't be started."""


def _setting(name, default=None):
    value = os.getenv(name, "").strip()
    return value or default


def _build_groq():
    if not _setting("GROQ_API_KEY"):
        raise LLMConfigError("Groq needs GROQ_API_KEY, but it is not set in .env.")
    model = _setting("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    log.info("Groq ready (model: %s)", model)
    return ChatGroq(model=model, temperature=0)


def _build_ollama():
    model = _setting("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    url = _setting("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL)
    try:
        pulled = [m.model for m in ollama.Client(host=url).list().models]
    except Exception as error:
        raise LLMConfigError(
            f"Can't reach Ollama at {url}. Is it running? ({error})"
        ) from error
    if model not in pulled:
        raise LLMConfigError(
            f"Ollama model '{model}' isn't pulled. Run: ollama pull {model}"
        )
    log.info("Ollama ready (model: %s)", model)
    # reasoning=False keeps qwen3's <think> text out of the reply.
    return ChatOllama(model=model, base_url=url, temperature=0, reasoning=False)


BUILDERS = {"groq": _build_groq, "ollama": _build_ollama}


def _read_provider(name, default, allowed):
    value = _setting(name, default).lower()
    if value not in allowed:
        raise LLMConfigError(
            f"{name}='{value}' is not valid. Use one of: {', '.join(allowed)}."
        )
    return value


def get_llm():
    """Start the provider named in LLM_PROVIDER, with LLM_FALLBACK as backup."""
    provider = _read_provider("LLM_PROVIDER", "groq", PROVIDERS)
    fallback = _read_provider("LLM_FALLBACK", "none", PROVIDERS + ("none",))

    llm = BUILDERS[provider]()

    if fallback in ("none", provider):
        return llm

    # A broken backup shouldn't stop a working main model from starting.
    try:
        backup = BUILDERS[fallback]()
    except LLMConfigError as error:
        log.warning("Fallback '%s' not available, running without it: %s", fallback, error)
        return llm

    backup = backup.with_listeners(
        on_start=lambda run: log.warning(
            "%s failed, the answer is coming from the %s fallback.", provider, fallback
        )
    )
    return llm.with_fallbacks([backup])
