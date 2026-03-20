"""
llm_hf.py – HuggingFace transformer-based LLM for local structured extraction.

Architecture role
-----------------
This implements the dedicated **Extraction Layer (LLM-1)** using an
encoder-decoder transformer (default: google/flan-t5-base).

Design rationale
----------------
- Flan-T5 is fine-tuned on instruction-following tasks, making it excellent
  for structured JSON extraction from JDs and resumes without requiring a
  running Ollama server or any API key.
- Encoder-decoder architecture (seq2seq) is more efficient than generative
  decoder-only models for extraction/classification tasks.
- Weights are downloaded once from HuggingFace Hub and cached locally.
- Lazy loading: model is not loaded until the first generate() call.
- CUDA is used automatically when available; CPU fallback is seamless.

Recommended models (trade-off: speed vs quality)
-------------------------------------------------
  google/flan-t5-base   ~250M params  fast,   good for short JDs
  google/flan-t5-large  ~780M params  medium, better entity coverage
  google/flan-t5-xl     ~3B params    slower, best extraction quality

Requires: pip install transformers torch
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional

from llm_base import BaseLLM

logger = logging.getLogger(__name__)


class HuggingFaceLLM(BaseLLM):
    """
    Extraction-layer LLM backed by a local HuggingFace seq2seq model.

    Uses Rotary Positional Embedding-compatible encoder-decoder architecture
    for accurate extraction across longer document segments.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID (default: google/flan-t5-base).
    max_input_tokens : int
        Maximum tokens fed to the encoder; longer inputs are truncated with
        a logged warning.  Increase for flan-t5-large/xl variants.
    max_new_tokens : int
        Maximum tokens the decoder may generate per call.
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        max_input_tokens: int = 512,
        max_new_tokens: int = 384,
    ):
        super().__init__(model_name)
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self._tokenizer = None
        self._model = None
        self._device: Optional[str] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Lazy-load tokenizer and model on first use."""
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "HuggingFace transformers and torch are required for the 'hf' provider. "
                "Install with: pip install transformers torch"
            )

        logger.info("Loading HuggingFace model '%s' …", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

        import torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = self._model.to(self._device)
        logger.info(
            "HuggingFace model '%s' loaded on %s", self.model_name, self._device
        )

    # ------------------------------------------------------------------
    # BaseLLM interface
    # ------------------------------------------------------------------

    def generate(self, prompt: str, temperature: float = 0.1) -> str:
        """
        Run the encoder-decoder model on *prompt* and return decoded text.

        Parameters
        ----------
        prompt : str
            Fully-rendered instruction prompt.
        temperature : float
            0.0 → greedy / beam search; >0.0 → sampling with given temperature.

        Returns
        -------
        str
            Decoded model output (plain text, typically JSON for extraction tasks).
        """
        self._load()

        import torch

        # Tokenise with truncation – warn when input exceeds the limit
        input_ids = self._tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_input_tokens,
            truncation=True,
            padding=False,
        ).input_ids

        if input_ids.shape[-1] == self.max_input_tokens:
            logger.warning(
                "Prompt truncated to %d tokens for model '%s'. "
                "Use flan-t5-large/xl or set HF_MAX_INPUT_TOKENS for better coverage.",
                self.max_input_tokens,
                self.model_name,
            )

        input_ids = input_ids.to(self._device)

        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            early_stopping=True,
        )
        if temperature > 0.0:
            gen_kwargs.update(do_sample=True, temperature=temperature)
        else:
            gen_kwargs.update(num_beams=4, do_sample=False)

        with torch.no_grad():
            output_ids = self._model.generate(input_ids, **gen_kwargs)

        return self._tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def is_available(self) -> bool:
        """
        Return True if the transformers + torch stack can be imported.

        Used by llm_factory to warn early when dependencies are missing.
        """
        try:
            import torch          # noqa: F401
            import transformers   # noqa: F401
            return True
        except ImportError:
            return False
