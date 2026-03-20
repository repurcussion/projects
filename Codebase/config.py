"""
config.py – Central configuration for the CV Sorting pipeline.
All model names, weights, and thresholds live here so they can be
swapped without touching any other module.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# LLM back-end selection
# ---------------------------------------------------------------------------

# Supported providers: "auto" | "ollama" | "openai" | "hf"
# auto = Ollama (local) → HuggingFace (local) [→ OpenAI if OPENAI_API_KEY set]
# Recommended for capstone: auto (Ollama-first, free, private, no API key needed)
LLM_PROVIDER: str = os.environ.get("LLM_PROVIDER", "auto")

# LLM-1  → lightweight parsing model
PARSER_MODEL: str = os.environ.get("PARSER_MODEL", "gemma:2b")

# LLM-2  → heavyweight reasoning / scoring model
SCORER_MODEL: str = os.environ.get("SCORER_MODEL", "llama3")

# Ollama base URL (local)
OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# OpenAI – read from environment, never hardcoded
OPENAI_API_KEY: Optional[str] = os.environ.get("OPENAI_API_KEY")
OPENAI_PARSER_MODEL: str = os.environ.get("OPENAI_PARSER_MODEL", "gpt-3.5-turbo")
OPENAI_SCORER_MODEL: str = os.environ.get("OPENAI_SCORER_MODEL", "gpt-4o")

# HuggingFace (local encoder-decoder, no server or API key required)
# LLM-1 extraction layer: Flan-T5 instruction-following seq2seq model
HF_PARSER_MODEL: str = os.environ.get("HF_PARSER_MODEL", "google/flan-t5-base")
HF_MAX_INPUT_TOKENS: int = int(os.environ.get("HF_MAX_INPUT_TOKENS", "512"))
HF_MAX_NEW_TOKENS: int = int(os.environ.get("HF_MAX_NEW_TOKENS", "384"))

# ---------------------------------------------------------------------------
# Scoring weights  (must sum to 1.0)
# 4-pillar schema: Technical (40%) + Experience (30%) + Soft Skills (15%) + Impact (15%)
# ---------------------------------------------------------------------------
WEIGHT_TECHNICAL: float = 0.40
WEIGHT_EXPERIENCE: float = 0.30
WEIGHT_SOFT_SKILLS: float = 0.15
WEIGHT_IMPACT: float = 0.15

# ---------------------------------------------------------------------------
# Hybrid matching blend
# keyword_weight + llm_weight must sum to 1.0
# ---------------------------------------------------------------------------
KEYWORD_WEIGHT: float = 0.30
LLM_WEIGHT: float = 0.70

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
TOP_K: int = 3          # k for Precision@K / Recall@K
RELEVANCE_THRESHOLD: float = 60.0   # score >= this → "relevant"

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
LLM_TIMEOUT: int = 120          # seconds per LLM call
MAX_RETRIES: int = 3            # retry attempts on LLM failure
OUTPUT_FILE: str = "results.json"

# ---------------------------------------------------------------------------
# Dataclass wrappers (optional typed access)
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    provider: str = LLM_PROVIDER
    parser_model: str = PARSER_MODEL
    scorer_model: str = SCORER_MODEL
    ollama_base_url: str = OLLAMA_BASE_URL
    openai_api_key: Optional[str] = OPENAI_API_KEY
    openai_parser_model: str = OPENAI_PARSER_MODEL
    openai_scorer_model: str = OPENAI_SCORER_MODEL
    hf_parser_model: str = HF_PARSER_MODEL
    hf_max_input_tokens: int = HF_MAX_INPUT_TOKENS
    hf_max_new_tokens: int = HF_MAX_NEW_TOKENS
    timeout: int = LLM_TIMEOUT
    max_retries: int = MAX_RETRIES


@dataclass
class ScoringConfig:
    weight_technical: float = WEIGHT_TECHNICAL
    weight_experience: float = WEIGHT_EXPERIENCE
    weight_soft_skills: float = WEIGHT_SOFT_SKILLS
    weight_impact: float = WEIGHT_IMPACT
    keyword_weight: float = KEYWORD_WEIGHT
    llm_weight: float = LLM_WEIGHT
    top_k: int = TOP_K
    relevance_threshold: float = RELEVANCE_THRESHOLD


# Singleton instances used across modules
llm_cfg = LLMConfig()
scoring_cfg = ScoringConfig()
