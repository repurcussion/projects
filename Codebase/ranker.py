"""
ranker.py – Sorts candidates by final_score and enriches results with rank metadata.

Takes a list of MatchResult objects from the matcher, sorts them descending
by final_score, and returns RankedCandidate objects that also carry:
  - 1-indexed rank position
  - Weighted component scores (skills 50%, experience 30%, education 20%)
  - Letter grade (A/B/C/D/F)
  - Human-readable recommendation (Strong Hire / Consider / Weak Consider / Reject)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from matcher import MatchResult   # flat import
from config import scoring_cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RankedCandidate – enriched result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RankedCandidate:
    """
    A MatchResult enriched with ranking position, weighted component scores,
    a letter grade, and a hiring recommendation.
    """
    rank: int                          # 1-indexed position in the leaderboard
    candidate_name: str
    final_score: float                 # blended hybrid score (0-100)

    # Weighted pillar contributions (raw 0-40/30/15/15 values from LLM)
    weighted_technical: float = 0.0      # technical pillar raw score (0-40)
    weighted_experience: float = 0.0     # experience pillar raw score (0-30)
    weighted_soft_skills: float = 0.0    # soft skills pillar raw score (0-15)
    weighted_impact: float = 0.0         # impact pillar raw score (0-15)

    # Normalised LLM sub-scores 0-100 (for metrics)
    keyword_score: float = 0.0
    llm_overall_score: float = 0.0
    llm_technical_score: float = 0.0
    llm_experience_score: float = 0.0
    llm_soft_skills_score: float = 0.0
    llm_impact_score: float = 0.0

    # Key matches and critical gaps from LLM
    matched_skills: List[str] = field(default_factory=list)
    missing_skills: List[str] = field(default_factory=list)
    key_matches: List[str] = field(default_factory=list)
    critical_gaps: List[str] = field(default_factory=list)
    verdict: str = ""

    # LLM-generated explanation of the score
    explanation: str = ""

    # Derived metadata
    grade: str = ""           # A / B / C / D / F
    recommendation: str = ""  # Strong Hire / Consider / Weak Consider / Reject

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict for JSON output and reporting."""
        return {
            "rank": self.rank,
            "candidate_name": self.candidate_name,
            "final_score": round(self.final_score, 2),
            "grade": self.grade,
            "recommendation": self.recommendation,
            "verdict": self.verdict,
            "weighted_technical": round(self.weighted_technical, 2),
            "weighted_experience": round(self.weighted_experience, 2),
            "weighted_soft_skills": round(self.weighted_soft_skills, 2),
            "weighted_impact": round(self.weighted_impact, 2),
            "keyword_score": round(self.keyword_score, 2),
            "llm_overall_score": round(self.llm_overall_score, 2),
            "llm_technical_score": round(self.llm_technical_score, 2),
            "llm_experience_score": round(self.llm_experience_score, 2),
            "llm_soft_skills_score": round(self.llm_soft_skills_score, 2),
            "llm_impact_score": round(self.llm_impact_score, 2),
            "key_matches": self.key_matches,
            "critical_gaps": self.critical_gaps,
            "matched_skills": self.matched_skills,
            "missing_skills": self.missing_skills,
            "explanation": self.explanation,
        }


# ---------------------------------------------------------------------------
# Grade and recommendation helpers
# ---------------------------------------------------------------------------

def _grade(score: float) -> str:
    """
    Convert a 0-100 numeric score to a letter grade.

    A: ≥85  |  B: ≥70  |  C: ≥55  |  D: ≥40  |  F: <40
    """
    if score >= 85:
        return "A"
    if score >= 70:
        return "B"
    if score >= 55:
        return "C"
    if score >= 40:
        return "D"
    return "F"


def _recommendation(score: float) -> str:
    """
    Map a numeric score to a human-readable hiring recommendation.

    ≥80 → Strong Hire  |  ≥60 → Hire  |  ≥40 → Potential  |  <40 → Reject
    """
    if score >= 80:
        return "Strong Hire"
    if score >= 60:
        return "Hire"
    if score >= 40:
        return "Potential"
    return "Reject"


# ---------------------------------------------------------------------------
# Ranker
# ---------------------------------------------------------------------------

class Ranker:
    """
    Sorts a list of MatchResults into a ranked leaderboard.

    Recomputes weighted pillar contributions using the configured weights so the
    breakdown is transparent and independent of how the matcher blended scores.

    Parameters
    ----------
    weight_technical : float
        Technical skills weight (default 0.40 from config).
    weight_experience : float
        Experience weight (default 0.30 from config).
    weight_soft_skills : float
        Soft skills & leadership weight (default 0.15 from config).
    weight_impact : float
        Project impact weight (default 0.15 from config).
    """

    def __init__(
        self,
        weight_technical: float = None,
        weight_experience: float = None,
        weight_soft_skills: float = None,
        weight_impact: float = None,
    ):
        # Use config defaults if not explicitly provided
        self.w_tech = weight_technical  or scoring_cfg.weight_technical
        self.w_exp  = weight_experience or scoring_cfg.weight_experience
        self.w_soft = weight_soft_skills or scoring_cfg.weight_soft_skills
        self.w_imp  = weight_impact      or scoring_cfg.weight_impact

    def rank(self, match_results: List[MatchResult]) -> List[RankedCandidate]:
        """
        Sort *match_results* by final_score descending and return
        a list of RankedCandidate objects with 1-indexed ranks.

        Ties in final_score are broken alphabetically by candidate name.

        Parameters
        ----------
        match_results : list of MatchResult

        Returns
        -------
        list of RankedCandidate  (sorted best → worst)
        """
        if not match_results:
            logger.warning("Ranker received empty match results.")
            return []

        # Sort descending by final_score; use name as tiebreaker
        sorted_results = sorted(
            match_results,
            key=lambda r: (-r.final_score, r.candidate_name),
        )

        ranked: List[RankedCandidate] = []
        for idx, result in enumerate(sorted_results, start=1):
            # Weighted pillar raw values (0-40/30/15/15) for breakdown display
            w_tech = self.w_tech * result.llm_technical_score
            w_exp  = self.w_exp  * result.llm_experience_score
            w_soft = self.w_soft * result.llm_soft_skills_score
            w_imp  = self.w_imp  * result.llm_impact_score

            rc = RankedCandidate(
                rank=idx,
                candidate_name=result.candidate_name,
                final_score=round(result.final_score, 2),
                weighted_technical=round(w_tech, 2),
                weighted_experience=round(w_exp, 2),
                weighted_soft_skills=round(w_soft, 2),
                weighted_impact=round(w_imp, 2),
                keyword_score=round(result.keyword_score, 2),
                llm_overall_score=round(result.llm_overall_score, 2),
                llm_technical_score=round(result.llm_technical_score, 2),
                llm_experience_score=round(result.llm_experience_score, 2),
                llm_soft_skills_score=round(result.llm_soft_skills_score, 2),
                llm_impact_score=round(result.llm_impact_score, 2),
                key_matches=result.key_matches,
                critical_gaps=result.critical_gaps,
                verdict=result.verdict,
                matched_skills=result.matched_skills,
                missing_skills=result.missing_skills,
                explanation=result.explanation,
                grade=_grade(result.final_score),
                recommendation=_recommendation(result.final_score),
            )
            ranked.append(rc)

        logger.info(
            "Ranked %d candidates. Top: %s (%.1f)",
            len(ranked),
            ranked[0].candidate_name,
            ranked[0].final_score,
        )
        return ranked
