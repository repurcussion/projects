"""
metrics.py – Evaluation metrics for the CV sorting pipeline.

Computed metrics
----------------
- Precision@K  : fraction of top-K candidates that are "relevant" (score ≥ threshold)
- Recall@K     : fraction of all relevant candidates found in top-K
- MRR          : Mean Reciprocal Rank – position of first relevant candidate
- NDCG@K       : Normalised Discounted Cumulative Gain at K
- Explainability Score : heuristic quality score for LLM explanations (0–1)
- Latency      : parse time, match time, total pipeline time
- Approach Comparison : keyword-only vs LLM-only vs hybrid mean/std scores
- Model Comparison : average LLM sub-scores across all candidates

Relevance is defined as final_score ≥ relevance_threshold (self-supervised proxy).
In a production system, relevance labels would come from human annotation.
"""

import logging
import math
import re
import time
from typing import Any, Dict, List

from ranker import RankedCandidate   # flat import
from config import scoring_cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Timer context manager (used in main.py to time pipeline stages)
# ---------------------------------------------------------------------------

class Timer:
    """
    Simple wall-clock timer implemented as a context manager.

    Usage:
        with Timer() as t:
            do_work()
        print(t.elapsed)   # seconds
    """

    def __init__(self):
        self._start: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self):
        """Record start time."""
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        """Compute elapsed time on exit."""
        self.elapsed = time.perf_counter() - self._start


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Computes all evaluation metrics for the CV sorting pipeline.

    Parameters
    ----------
    top_k : int
        K value for Precision@K and Recall@K (default from config).
    relevance_threshold : float
        Minimum final_score (0-100) for a candidate to be considered "relevant".
        Used as a proxy for ground-truth labels (self-supervised evaluation).
    """

    def __init__(
        self,
        top_k: int = None,
        relevance_threshold: float = None,
    ):
        self.top_k = top_k or scoring_cfg.top_k
        self.relevance_threshold = (
            relevance_threshold or scoring_cfg.relevance_threshold
        )

    def compute_metrics(
        self,
        ranked: List[RankedCandidate],
        parse_time: float = 0.0,
        match_time: float = 0.0,
        total_time: float = 0.0,
        parser_model: str = "unknown",
        scorer_model: str = "unknown",
        provider: str = "unknown",
    ) -> Dict[str, Any]:
        """
        Compute and return all evaluation metrics as a single dict.

        Parameters
        ----------
        ranked : list of RankedCandidate
            Ranked output from the Ranker.
        parse_time, match_time, total_time : float
            Wall-clock times in seconds from main.py timers.
        parser_model, scorer_model, provider : str
            Model metadata embedded in the output for reference.

        Returns
        -------
        dict with keys: core_metrics, latency, approach_comparison,
                        model_comparison, model_info
        """
        if not ranked:
            logger.warning("Evaluator received empty ranked list.")
            return {}

        # --- Define relevance set ---
        # Candidates with final_score >= threshold are "relevant" (proxy labels).
        # In production, replace with human annotation labels.
        relevant_set = {
            rc.candidate_name
            for rc in ranked
            if rc.final_score >= self.relevance_threshold
        }
        total_relevant = max(len(relevant_set), 1)   # avoid division by zero

        # --- Precision@K ---
        # Fraction of top-K candidates that are relevant
        top_k_names = [rc.candidate_name for rc in ranked[: self.top_k]]
        relevant_in_topk = sum(1 for n in top_k_names if n in relevant_set)
        precision_at_k = relevant_in_topk / self.top_k

        # --- Recall@K ---
        # Fraction of all relevant candidates that appear in the top-K
        recall_at_k = relevant_in_topk / total_relevant

        # --- Mean Reciprocal Rank (MRR) ---
        mrr = self._mean_reciprocal_rank(ranked, relevant_set)

        # --- NDCG@K ---
        ndcg_at_k = self._ndcg(ranked, relevant_set, self.top_k)

        # --- Explainability score ---
        # Heuristic proxy: checks explanation length, specificity, and completeness
        expl_score = self._explainability_score(ranked)

        # --- Approach comparison ---
        # Compares keyword-only, LLM-only, and hybrid scoring distributions
        approach_comparison = self._approach_comparison(ranked)

        # --- Model comparison ---
        # Average LLM sub-scores across all candidates
        model_comparison = self._model_comparison(ranked)

        # Build core metrics dict
        core_metrics = {
            f"precision_at_{self.top_k}": round(precision_at_k, 4),
            f"recall_at_{self.top_k}": round(recall_at_k, 4),
            "mrr": round(mrr, 4),
            f"ndcg_at_{self.top_k}": round(ndcg_at_k, 4),
            "explainability_score": round(expl_score, 4),
            "total_relevant_candidates": total_relevant,
            "relevance_threshold": self.relevance_threshold,
            "top_k": self.top_k,
        }

        # Pipeline stage latency
        latency = {
            "parse_time": round(parse_time, 3),
            "match_time": round(match_time, 3),
            "total_time": round(total_time, 3),
            "avg_time_per_candidate": round(
                match_time / max(len(ranked), 1), 3
            ),
        }

        # Model metadata for the results.json file
        model_info = {
            "provider": provider,
            "parser_model": parser_model,
            "scorer_model": scorer_model,
            "llm1_role": "Resume + JD Parsing",
            "llm2_role": "Semantic Scoring + Explanation",
        }

        return {
            "core_metrics": core_metrics,
            "latency": latency,
            "approach_comparison": approach_comparison,
            "model_comparison": model_comparison,
            "model_info": model_info,
        }

    # ------------------------------------------------------------------
    # Individual metric implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _mean_reciprocal_rank(
        ranked: List[RankedCandidate], relevant_set: set
    ) -> float:
        """
        Compute Mean Reciprocal Rank (MRR).

        MRR = 1 / rank_of_first_relevant_candidate.
        Returns 0.0 if no relevant candidate is found in the ranked list.
        Higher is better (maximum = 1.0 when the top candidate is relevant).
        """
        for rc in ranked:
            if rc.candidate_name in relevant_set:
                return 1.0 / rc.rank
        return 0.0

    @staticmethod
    def _ndcg(
        ranked: List[RankedCandidate], relevant_set: set, k: int
    ) -> float:
        """
        Compute NDCG@K using binary relevance judgements.

        DCG@K  = Σ rel_i / log2(i+2)  for i in 0..K-1
        NDCG@K = DCG@K / IDCG@K       (IDCG = ideal ordering)

        Returns 0.0 if there are no relevant candidates in the top-K.
        """
        def dcg(hits: List[int]) -> float:
            """Discounted Cumulative Gain for a binary relevance list."""
            return sum(rel / math.log2(i + 2) for i, rel in enumerate(hits))

        # Build binary relevance vector for top-K results
        actual_hits = [
            1 if rc.candidate_name in relevant_set else 0
            for rc in ranked[:k]
        ]
        # Ideal ordering: all relevant candidates first
        ideal_hits = sorted(actual_hits, reverse=True)

        idcg = dcg(ideal_hits)
        return dcg(actual_hits) / idcg if idcg > 0 else 0.0

    @staticmethod
    def _explainability_score(ranked: List[RankedCandidate]) -> float:
        """
        Compute a proxy explainability score based on explanation quality.

        Heuristic scoring per candidate (0.0 – 1.0):
        +0.4  explanation is non-empty
        +0.3  explanation contains a numeric score reference
        +0.2  explanation is ≥ 20 words
        +0.1  candidate has missing skills (shows the model identified gaps)

        Returns the average across all candidates.
        """
        if not ranked:
            return 0.0

        scores = []
        for rc in ranked:
            s = 0.0
            expl = rc.explanation or ""
            if expl.strip():
                s += 0.4   # explanation present
            if re.search(r"\d+", expl):
                s += 0.3   # contains a number (score reference)
            if len(expl.split()) >= 20:
                s += 0.2   # sufficiently detailed
            if rc.missing_skills:
                s += 0.1   # gaps identified
            scores.append(min(s, 1.0))

        return sum(scores) / len(scores)

    @staticmethod
    def _approach_comparison(
        ranked: List[RankedCandidate],
    ) -> Dict[str, float]:
        """
        Compare three ranking approaches for the same candidate set:

        - Keyword-only : uses keyword_score alone
        - LLM-only     : uses weighted LLM sub-scores (50%/30%/20%)
        - Hybrid       : uses final_score (30% keyword + 70% LLM)

        Reports mean and standard deviation for each approach.
        A lower std means more consistent/confident scores.
        """
        def _mean(scores):
            return sum(scores) / len(scores) if scores else 0.0

        kw_scores = [rc.keyword_score for rc in ranked]

        # LLM-only score: weighted combination of sub-scores
        llm_scores = [
            rc.llm_skills_score * 0.5
            + rc.llm_experience_score * 0.3
            + rc.llm_education_score * 0.2
            for rc in ranked
        ]

        hybrid_scores = [rc.final_score for rc in ranked]

        return {
            "keyword_only_mean": round(_mean(kw_scores), 2),
            "llm_only_mean": round(_mean(llm_scores), 2),
            "hybrid_mean": round(_mean(hybrid_scores), 2),
            "keyword_std": round(_std(kw_scores), 2),
            "llm_std": round(_std(llm_scores), 2),
            "hybrid_std": round(_std(hybrid_scores), 2),
        }

    @staticmethod
    def _model_comparison(ranked: List[RankedCandidate]) -> Dict[str, Any]:
        """
        Breakdown of average LLM sub-scores across all candidates.

        Useful for analysing model behaviour and identifying systematic biases
        (e.g. always scoring education higher than experience).
        """
        if not ranked:
            return {}

        def avg(lst):
            """Compute arithmetic mean, rounded to 2 decimal places."""
            return round(sum(lst) / len(lst), 2)

        return {
            "avg_llm_skills_score": avg([r.llm_skills_score for r in ranked]),
            "avg_llm_experience_score": avg([r.llm_experience_score for r in ranked]),
            "avg_llm_education_score": avg([r.llm_education_score for r in ranked]),
            "avg_keyword_score": avg([r.keyword_score for r in ranked]),
            "avg_final_score": avg([r.final_score for r in ranked]),
            "score_spread": round(
                max(r.final_score for r in ranked) - min(r.final_score for r in ranked), 2
            ),
        }


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _std(values: List[float]) -> float:
    """
    Compute population standard deviation.

    Returns 0.0 if fewer than 2 values are provided.
    """
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return variance ** 0.5
