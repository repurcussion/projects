"""
llm_fallback.py – Cascading fallback LLM that tries providers in priority order.

Default priority chain
----------------------
  1. OpenAI API  (cloud)  – fastest / highest quality; requires OPENAI_API_KEY
  2. Ollama      (local)  – full-quality local; requires ollama server + model
  3. HuggingFace (local)  – offline fallback; requires transformers + torch

Behaviour
---------
- Each provider is attempted in sequence.
- If a provider raises *any* exception (missing API key, network failure,
  connection refused, ImportError) it is logged as a WARNING and skipped.
- The first provider that returns a non-empty string is used for that call.
- If all providers fail, an empty string is returned and an ERROR is logged.
- The provider that succeeded is logged at DEBUG level for observability.

The FallbackLLM is transparent to the rest of the pipeline: it implements
the same BaseLLM.generate() interface as every other backend.
"""

import logging
from typing import List

from llm_base import BaseLLM

logger = logging.getLogger(__name__)


class FallbackLLM(BaseLLM):
    """
    Meta-LLM that cascades through a prioritised list of concrete LLMs.

    Parameters
    ----------
    llms : list of BaseLLM
        Ordered list of backends to try.  First entry has highest priority.
    """

    def __init__(self, llms: List[BaseLLM]):
        # Use a composite name for logging / repr
        names = " → ".join(llm.model_name for llm in llms)
        super().__init__(f"fallback({names})")
        self._llms = llms

    # ------------------------------------------------------------------
    # BaseLLM interface
    # ------------------------------------------------------------------

    def generate(self, prompt: str, temperature: float = 0.1) -> str:
        """
        Try each backend in order; return the first successful response.

        A backend is skipped if it raises any exception.  An empty string
        returned by a backend is treated as a soft failure and the next
        backend is tried.
        """
        for llm in self._llms:
            try:
                logger.debug("FallbackLLM trying: %s", llm.model_name)
                result = llm.generate(prompt, temperature)
                if result and result.strip():
                    logger.debug("FallbackLLM succeeded with: %s", llm.model_name)
                    return result
                logger.warning(
                    "FallbackLLM: '%s' returned empty response – trying next.",
                    llm.model_name,
                )
            except Exception as exc:
                logger.warning(
                    "FallbackLLM: '%s' failed (%s: %s) – trying next provider.",
                    llm.model_name,
                    type(exc).__name__,
                    exc,
                )

        logger.error(
            "FallbackLLM: all providers exhausted. Chain: %s", self.model_name
        )
        return ""
