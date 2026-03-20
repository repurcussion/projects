"""
llm_ollama.py – Ollama local LLM implementation.

Uses HTTP calls to a locally-running Ollama server to run
models like gemma:2b (LLM-1 parser) and llama3 (LLM-2 scorer).

Setup:
    1. Install Ollama: https://ollama.com/download
    2. Pull models:    ollama pull gemma:2b && ollama pull llama3
    3. Start server:   ollama serve  (or brew services start ollama)
"""

import time
import logging
from typing import Optional

import requests

from llm_base import BaseLLM   # flat import
from config import llm_cfg

logger = logging.getLogger(__name__)


class OllamaLLM(BaseLLM):
    """
    Calls a locally-running Ollama server via its REST API.

    Parameters
    ----------
    model_name : str
        Ollama model tag, e.g. "gemma:2b" or "llama3".
    base_url : str, optional
        Base URL of the Ollama API (default from config).
    timeout : int, optional
        Request timeout in seconds (default from config).
    max_retries : int, optional
        Number of retry attempts on transient failures.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = None,
        timeout: int = None,
        max_retries: int = None,
    ):
        super().__init__(model_name)
        # Use config defaults if not explicitly provided
        self.base_url = (base_url or llm_cfg.ollama_base_url).rstrip("/")
        self.timeout = timeout or llm_cfg.timeout
        self.max_retries = max_retries or llm_cfg.max_retries
        self._generate_url = f"{self.base_url}/api/generate"

    # ------------------------------------------------------------------
    # Public interface (implements BaseLLM.generate)
    # ------------------------------------------------------------------

    def generate(self, prompt: str, temperature: float = 0.1) -> str:
        """
        Send *prompt* to the Ollama /api/generate endpoint.

        Retries up to `max_retries` times on connection / timeout errors.
        Uses `num_predict=2048` to allow sufficient output for JSON responses.
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,            # get full response at once
            "options": {
                "temperature": temperature,
                "num_predict": 2048,    # enough tokens for complete JSON
            },
        }

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(
                    "OllamaLLM [%s] attempt %d/%d",
                    self.model_name, attempt, self.max_retries,
                )
                response = requests.post(
                    self._generate_url,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                return data.get("response", "").strip()

            except requests.exceptions.ConnectionError as exc:
                # Ollama server not reachable – retry with back-off
                last_error = exc
                logger.warning(
                    "Ollama connection failed (attempt %d): %s", attempt, exc
                )
                time.sleep(2 * attempt)

            except requests.exceptions.Timeout as exc:
                # Request timed out – retry with back-off
                last_error = exc
                logger.warning(
                    "Ollama request timed out (attempt %d): %s", attempt, exc
                )
                time.sleep(2 * attempt)

            except requests.exceptions.HTTPError as exc:
                # Non-transient HTTP error (e.g. 404 model not found) – no retry
                last_error = exc
                logger.error("Ollama HTTP error: %s", exc)
                break

        raise RuntimeError(
            f"OllamaLLM '{self.model_name}' failed after {self.max_retries} "
            f"attempts. Last error: {last_error}"
        )

    # ------------------------------------------------------------------
    # Health check helper
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """
        Return True if the Ollama server is reachable and the model is loaded.
        Used at startup to warn the user before long API calls fail.
        """
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code != 200:
                return False
            models = [m["name"] for m in resp.json().get("models", [])]
            # Accept prefix match (e.g. "gemma:2b" matches "gemma:2b-instruct")
            return any(self.model_name.split(":")[0] in m for m in models)
        except Exception:
            return False
