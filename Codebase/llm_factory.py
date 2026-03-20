"""
llm_factory.py – Factory functions that wire config → concrete LLM instances.

Centralising construction here means the rest of the pipeline never
has to import concrete LLM classes directly.  Swapping the backend
requires only a config/CLI change, not a code change.

Two public functions:
    get_parser_llm()  → LLM-1: lightweight model for parsing (gemma:2b / gpt-3.5-turbo)
    get_scorer_llm()  → LLM-2: reasoning model for scoring (llama3 / gpt-4o)
"""

import logging

from llm_base import BaseLLM        # flat imports
from llm_ollama import OllamaLLM
from llm_openai import OpenAILLM
from config import llm_cfg

logger = logging.getLogger(__name__)


def _build_llm(model_name: str, openai_model: str) -> BaseLLM:
    """
    Instantiate the appropriate LLM backend based on LLM_PROVIDER config.

    Parameters
    ----------
    model_name : str
        Ollama model tag (e.g. "gemma:2b") – used when provider == "ollama".
    openai_model : str
        OpenAI model name (e.g. "gpt-3.5-turbo") – used when provider == "openai".

    Returns
    -------
    BaseLLM
        Concrete LLM instance ready for use.

    Raises
    ------
    ValueError
        If LLM_PROVIDER is set to an unsupported value.
    """
    provider = llm_cfg.provider.lower()

    if provider == "ollama":
        llm = OllamaLLM(model_name=model_name)
        # Warn if the model is not yet pulled locally
        if not llm.is_available():
            logger.warning(
                "Ollama model '%s' not found locally. "
                "Pull it with: ollama pull %s",
                model_name, model_name,
            )
        return llm

    elif provider == "openai":
        # API key is read from OPENAI_API_KEY env var inside OpenAILLM
        return OpenAILLM(model_name=openai_model)

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}'. "
            "Set LLM_PROVIDER to 'ollama' or 'openai'."
        )


def get_parser_llm() -> BaseLLM:
    """
    Return LLM-1: the lightweight parsing model.

    Ollama → gemma:2b  |  OpenAI → gpt-3.5-turbo
    Used to extract structured JSON from raw resume and JD text.
    """
    logger.info(
        "Instantiating parser LLM: provider=%s model=%s",
        llm_cfg.provider,
        llm_cfg.parser_model,
    )
    return _build_llm(llm_cfg.parser_model, llm_cfg.openai_parser_model)


def get_scorer_llm() -> BaseLLM:
    """
    Return LLM-2: the heavyweight scoring / reasoning model.

    Ollama → llama3  |  OpenAI → gpt-4o
    Used to semantically score candidates and generate explanations.
    """
    logger.info(
        "Instantiating scorer LLM: provider=%s model=%s",
        llm_cfg.provider,
        llm_cfg.scorer_model,
    )
    return _build_llm(llm_cfg.scorer_model, llm_cfg.openai_scorer_model)
