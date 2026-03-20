"""
llm_openai.py – OpenAI LLM implementation (optional back-end).

Set the environment variable OPENAI_API_KEY before using this class.
The API key is NEVER hardcoded; it is always read from the environment
or passed explicitly at construction time.

Usage:
    export OPENAI_API_KEY="sk-..."
    python main.py --resumes ./resumes --jd jd.txt \\
        --provider openai --parser gpt-3.5-turbo --scorer gpt-4o
"""

import logging
import time
from typing import Optional

from llm_base import BaseLLM   # flat import
from config import llm_cfg

logger = logging.getLogger(__name__)


class OpenAILLM(BaseLLM):
    """
    Wraps the OpenAI chat-completions endpoint.

    Parameters
    ----------
    model_name : str
        OpenAI model identifier, e.g. "gpt-3.5-turbo" or "gpt-4o".
    api_key : str, optional
        Overrides the OPENAI_API_KEY environment variable.
        If not provided, reads from environment (recommended).
    max_retries : int, optional
        Number of retry attempts for transient API errors.
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        max_retries: int = None,
    ):
        super().__init__(model_name)
        # Read API key from argument or environment – never hardcoded
        self.api_key = api_key or llm_cfg.openai_api_key
        self.max_retries = max_retries or llm_cfg.max_retries

        if not self.api_key:
            raise EnvironmentError(
                "OpenAI API key not found. "
                "Set the OPENAI_API_KEY environment variable or pass api_key=."
            )

        # Lazy import: openai package only required when using this backend
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for OpenAILLM. "
                "Install it with: pip install openai"
            ) from exc

    def generate(self, prompt: str, temperature: float = 0.1) -> str:
        """
        Send *prompt* to the OpenAI chat completions API.

        Retries on transient errors with exponential back-off.
        """
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(
                    "OpenAILLM [%s] attempt %d/%d",
                    self.model_name, attempt, self.max_retries,
                )
                # Use chat completions – works for all current OpenAI models
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=1024,
                )
                return response.choices[0].message.content.strip()

            except Exception as exc:  # noqa: BLE001 – catch all API errors
                last_error = exc
                logger.warning(
                    "OpenAI API error (attempt %d): %s", attempt, exc
                )
                time.sleep(2 * attempt)   # exponential back-off

        raise RuntimeError(
            f"OpenAILLM '{self.model_name}' failed after {self.max_retries} "
            f"attempts. Last error: {last_error}"
        )
