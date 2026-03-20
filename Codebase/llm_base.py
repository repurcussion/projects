"""
llm_base.py – Abstract base class for all LLM implementations.

Every concrete LLM (OllamaLLM, OpenAILLM) must implement the
`generate(prompt) -> str` method defined here.  The rest of the
pipeline depends ONLY on this interface, making model swaps a
one-line config change.
"""

from abc import ABC, abstractmethod
from typing import Optional


class BaseLLM(ABC):
    """
    Minimal interface that every LLM backend must honour.

    Attributes
    ----------
    model_name : str
        Human-readable identifier used in logging and reports.
    """

    def __init__(self, model_name: str):
        # Store model name for logging / repr
        self.model_name = model_name

    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.1) -> str:
        """
        Send *prompt* to the underlying model and return the raw text reply.

        Parameters
        ----------
        prompt : str
            The complete, fully-rendered prompt string.
        temperature : float
            Sampling temperature (lower = more deterministic).

        Returns
        -------
        str
            Raw model output (may contain JSON, markdown, or plain text).
        """

    def __repr__(self) -> str:
        """Return a readable representation showing class and model name."""
        return f"{self.__class__.__name__}(model={self.model_name!r})"
