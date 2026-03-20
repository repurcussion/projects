"""
llm_factory.py – Factory functions that wire config → concrete LLM instances.

Centralising construction here means the rest of the pipeline never
has to import concrete LLM classes directly.  Swapping the backend
requires only a config/CLI change, not a code change.

Two public functions:
    get_parser_llm()  → LLM-1: extraction layer
    get_scorer_llm()  → LLM-2: reasoning / scoring model

Default provider is 'auto': OpenAI API → Ollama (local) → HuggingFace (local)
"""

import logging

from llm_base import BaseLLM        # flat imports
from llm_ollama import OllamaLLM
from llm_openai import OpenAILLM
from llm_hf import HuggingFaceLLM
from llm_fallback import FallbackLLM
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

    elif provider == "hf":
        # Local HuggingFace encoder-decoder model (no server, no API key)
        hf_llm = HuggingFaceLLM(
            model_name=llm_cfg.hf_parser_model,
            max_input_tokens=llm_cfg.hf_max_input_tokens,
            max_new_tokens=llm_cfg.hf_max_new_tokens,
        )
        if not hf_llm.is_available():
            logger.warning(
                "transformers/torch not installed. "
                "Install with: pip install transformers torch"
            )
        return hf_llm

    elif provider == "auto":
        # Cascading fallback: OpenAI API → Ollama → HuggingFace
        # Build each backend defensively; skip any that fail at construction
        # (e.g. missing API key, missing package) and log a warning.
        _candidates = [
            ("OpenAI",       lambda: OpenAILLM(model_name=openai_model)),
            ("Ollama",       lambda: OllamaLLM(model_name=model_name)),
            ("HuggingFace",  lambda: HuggingFaceLLM(
                model_name=llm_cfg.hf_parser_model,
                max_input_tokens=llm_cfg.hf_max_input_tokens,
                max_new_tokens=llm_cfg.hf_max_new_tokens,
            )),
        ]
        chain = []
        for label, factory in _candidates:
            try:
                chain.append(factory())
            except Exception as exc:
                logger.warning(
                    "Provider=auto: %s unavailable at init (%s: %s) – skipping.",
                    label, type(exc).__name__, exc,
                )
        if not chain:
            raise RuntimeError(
                "Provider=auto: no LLM backends are available. "
                "Set OPENAI_API_KEY, start Ollama, or install transformers+torch."
            )
        logger.info(
            "Provider=auto: chain = %s",
            " → ".join(llm.model_name for llm in chain),
        )
        return FallbackLLM(chain)

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}'. "
            "Set LLM_PROVIDER to 'auto', 'ollama', 'openai', or 'hf'."
        )


def get_parser_llm() -> BaseLLM:
    """
    Return LLM-1: the extraction layer model.

    Ollama  → gemma:2b          (local, requires ollama pull gemma:2b)
    OpenAI  → gpt-3.5-turbo     (cloud, requires OPENAI_API_KEY)
    HF      → flan-t5-base      (local, downloaded from HuggingFace Hub)

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
