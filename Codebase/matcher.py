"""
matcher.py – Hybrid candidate-job matching engine (LLM-2 + keyword overlap).

Two-step scoring pipeline:
    Step 1 – Deterministic keyword overlap (fast, interpretable, always runs)
    Step 2 – LLM-2 semantic scoring      (accurate, nuanced, may fail gracefully)
    Step 3 – Weighted blend: 30% keyword + 70% LLM  (configurable via config.py)

Compound JD skill strings like "PyTorch or TensorFlow" or "Docker/Kubernetes"
are automatically split into individual tokens before keyword matching.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# LangChain v1.x: prefer langchain_core, fall back to legacy path
try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    from langchain.prompts import PromptTemplate

from llm_base import BaseLLM                          # flat imports
from parser import ParsedResume, ParsedJD, _extract_json, _safe_float
from config import scoring_cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MatchResult dataclass – all scoring details for one candidate
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    """
    Stores all scoring details for a single candidate against a single JD.

    Component scores are 0-100.  final_score is the weighted blend.
    """
    candidate_name: str

    # Keyword overlap score (Step 1) – 0 to 100
    keyword_score: float = 0.0

    # LLM pillar scores (Step 2) – normalised 0 to 100 each
    llm_technical_score: float = 0.0      # Core Technical Skills (40%)
    llm_experience_score: float = 0.0     # Relevant Experience (30%)
    llm_soft_skills_score: float = 0.0    # Soft Skills & Leadership (15%)
    llm_impact_score: float = 0.0         # Project Impact (15%)
    llm_overall_score: float = 0.0        # overall_fit_score from LLM

    # Final blended score (Step 3)
    final_score: float = 0.0

    # Human-readable reasoning produced by llama3
    explanation: str = ""

    # Keyword matching details (from Step 1) and LLM-identified gaps
    matched_skills: List[str] = field(default_factory=list)
    missing_skills: List[str] = field(default_factory=list)
    key_matches: List[str] = field(default_factory=list)
    critical_gaps: List[str] = field(default_factory=list)
    verdict: str = ""

    # Error from LLM scoring (None if scoring succeeded)
    llm_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict for JSON output."""
        return {
            "candidate_name": self.candidate_name,
            "final_score": round(self.final_score, 2),
            "keyword_score": round(self.keyword_score, 2),
            "llm_technical_score": round(self.llm_technical_score, 2),
            "llm_experience_score": round(self.llm_experience_score, 2),
            "llm_soft_skills_score": round(self.llm_soft_skills_score, 2),
            "llm_impact_score": round(self.llm_impact_score, 2),
            "llm_overall_score": round(self.llm_overall_score, 2),
            "key_matches": self.key_matches,
            "critical_gaps": self.critical_gaps,
            "verdict": self.verdict,
            "matched_skills": self.matched_skills,
            "missing_skills": self.missing_skills,
            "explanation": self.explanation,
            "llm_error": self.llm_error,
        }


# ---------------------------------------------------------------------------
# Scoring prompt for LLM-2 (llama3 / gpt-4o)
# ---------------------------------------------------------------------------

SCORE_TEMPLATE = """\
### ROLE
You are an expert Technical Recruiter with 20 years of experience in specialised talent
acquisition. Evaluate the Candidate Resume against the Job Description with high precision.

### INPUT DATA
1. **Job Description (JD):** {parsed_jd_data}
2. **Candidate Resume:** {structured_resume_data}

### EVALUATION CRITERIA
Evaluate alignment across four pillars:
1. **Core Technical Skills (40%):** Does the candidate possess the mandatory tech stack?
2. **Relevant Experience (30%):** Years of experience and domain expertise match?
3. **Soft Skills & Leadership (15%):** Evidence of communication, team management, problem-solving.
4. **Project Impact (15%):** Quality of achievements ("Reduced latency by 20%" vs "Worked on latency").

### INSTRUCTIONS
- Use Chain-of-Thought reasoning: extract evidence that matches or conflicts with the JD.
- Be critical. If a skill is mentioned but no practical application is shown, give a partial score.
- Identify Missing Critical Gaps that might be a deal-breaker.
- Do NOT assume skills not explicitly listed. If data is missing, treat it as not present.
- ANTI-ANCHORING: Every candidate is unique. Score each pillar independently on specific
  evidence. Do NOT assign the same score to different candidates.
- Use PRECISE values, not round numbers. Avoid multiples of 5 or 10 unless the math demands it.
- CONSTRAINT: overall_fit_score MUST equal exactly:
  pillar_scores.technical + pillar_scores.experience + pillar_scores.soft_skills + pillar_scores.impact

### OUTPUT FORMAT
Respond ONLY with valid JSON (no markdown, no extra text):
{{
  "candidate_name": "{candidate_name}",
  "overall_fit_score": <MUST equal technical + experience + soft_skills + impact>,
  "justification": "<3-sentence summary of the decision>",
  "pillar_scores": {{
    "technical": <integer 0-40>,
    "experience": <integer 0-30>,
    "soft_skills": <integer 0-15>,
    "impact": <integer 0-15>
  }},
  "key_matches": ["<top 3 matching skills/experiences>"],
  "critical_gaps": ["<missing mandatory requirements>"],
  "verdict": "<Strong Hire / Hire / Potential / Reject>"
}}
"""


# ---------------------------------------------------------------------------
# HybridMatcher
# ---------------------------------------------------------------------------

class HybridMatcher:
    """
    Scores a candidate against a job description using a hybrid approach.

    Step 1 – keyword overlap (deterministic):
        Required/preferred skills from the JD are matched against candidate
        skills using exact, substring, and acronym matching.

    Step 2 – LLM semantic scoring (LLM-2):
        A structured prompt is sent to LLM-2, which returns numeric scores
        for skills (50%), experience (30%), and education (20%), plus an
        explanation. On failure, keyword score is used as fallback.

    Step 3 – weighted blend:
        final_score = keyword_weight × keyword_score + llm_weight × llm_overall

    Parameters
    ----------
    scorer_llm : BaseLLM
        LLM-2: the reasoning/scoring model (e.g. llama3 via OllamaLLM).
    keyword_weight : float
        Fraction of final score from keyword overlap (default 0.30).
    llm_weight : float
        Fraction of final score from LLM scoring (default 0.70).
    """

    def __init__(
        self,
        scorer_llm: BaseLLM,
        keyword_weight: float = None,
        llm_weight: float = None,
    ):
        self.scorer_llm = scorer_llm
        # Fall back to config values if not explicitly set
        self.keyword_weight = keyword_weight or scoring_cfg.keyword_weight
        self.llm_weight = llm_weight or scoring_cfg.llm_weight

        # Build the LangChain prompt template for llama3 scoring
        self._prompt = PromptTemplate(
            input_variables=[
                "candidate_name", "parsed_jd_data", "structured_resume_data",
            ],
            template=SCORE_TEMPLATE,
        )

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def match(self, resume: ParsedResume, jd: ParsedJD) -> MatchResult:
        """
        Score *resume* against *jd* and return a MatchResult.

        Keyword score is always computed first.  LLM score is computed second;
        on LLM failure the keyword score is used as the sole signal.
        """
        result = MatchResult(candidate_name=resume.candidate_name)

        # Step 1 – deterministic keyword overlap
        kw_score, matched, missing = self._keyword_score(resume, jd)
        result.keyword_score = kw_score
        result.matched_skills = matched
        result.missing_skills = missing

        # Step 2 – LLM semantic scoring
        llm_result = self._llm_score(resume, jd)

        if llm_result is not None:
            pillars = llm_result.get("pillar_scores", {})
            raw_tech  = _safe_float(pillars.get("technical",   0))
            raw_exp   = _safe_float(pillars.get("experience",  0))
            raw_soft  = _safe_float(pillars.get("soft_skills", 0))
            raw_imp   = _safe_float(pillars.get("impact",      0))
            # Normalise each pillar to 0-100 for uniform sub-score display
            result.llm_technical_score   = min(raw_tech  / 0.40, 100.0)
            result.llm_experience_score  = min(raw_exp   / 0.30, 100.0)
            result.llm_soft_skills_score = min(raw_soft  / 0.15, 100.0)
            result.llm_impact_score      = min(raw_imp   / 0.15, 100.0)
            # Compute overall from raw pillar sum – guaranteed consistent, overrides LLM anchor
            result.llm_overall_score = min(raw_tech + raw_exp + raw_soft + raw_imp, 100.0)
            result.explanation = llm_result.get("justification", "")
            result.key_matches = llm_result.get("key_matches", [])[:3]
            result.critical_gaps = llm_result.get("critical_gaps", [])
            result.verdict = llm_result.get("verdict", "")
        else:
            # LLM failed – fall back to keyword score for all pillars
            exp_score = self._experience_heuristic(resume, jd)
            result.llm_technical_score   = kw_score
            result.llm_experience_score  = exp_score
            result.llm_soft_skills_score = 50.0   # neutral when unknown
            result.llm_impact_score      = 50.0   # neutral when unknown
            result.llm_overall_score     = kw_score
            result.llm_error = "LLM scoring failed – keyword fallback used"
            result.explanation = (
                f"LLM scoring unavailable. "
                f"Keyword overlap: {kw_score:.1f}/100. "
                f"Matched skills: {', '.join(matched) if matched else 'none'}."
            )

        # Step 3 – weighted blend of keyword and LLM scores
        result.final_score = (
            self.keyword_weight * result.keyword_score
            + self.llm_weight * result.llm_overall_score
        )
        result.final_score = max(0.0, min(100.0, result.final_score))  # clamp to [0,100]

        logger.info(
            "Matched '%s' → final=%.1f (kw=%.1f, llm=%.1f)",
            resume.candidate_name,
            result.final_score,
            result.keyword_score,
            result.llm_overall_score,
        )
        return result

    def match_batch(
        self, resumes: List[ParsedResume], jd: ParsedJD
    ) -> List[MatchResult]:
        """
        Score all *resumes* against *jd* and return a list of MatchResults.

        Processes candidates sequentially to avoid Ollama overload.
        """
        return [self.match(r, jd) for r in resumes]

    # ------------------------------------------------------------------
    # Step 1 – Keyword overlap scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _expand_skills(skills: List[str]) -> List[str]:
        """
        Expand compound skill strings into individual tokens.

        Examples:
            "PyTorch or TensorFlow"    → ["pytorch", "tensorflow"]
            "Docker/Kubernetes"        → ["docker", "kubernetes"]
            "AWS and GCP"              → ["aws", "gcp"]
        """
        expanded = []
        for skill in skills:
            # Split on " or ", " and ", or "/" separators
            parts = re.split(r"\s+or\s+|\s+and\s+|/", skill, flags=re.IGNORECASE)
            expanded.extend(p.strip() for p in parts if p.strip())
        return expanded

    def _keyword_score(
        self, resume: ParsedResume, jd: ParsedJD
    ):
        """
        Compute normalised keyword overlap score between resume and JD skills.

        Scoring formula:
            required_score  = (required_matched / total_required) × 100  → 70% weight
            preferred_score = (preferred_matched / total_preferred) × 100 → 30% weight
            kw_score = 0.70 × required_score + 0.30 × preferred_score

        Returns
        -------
        tuple (score: float, matched: list[str], missing: list[str])
        """
        # Expand compound skill strings before matching
        required = self._expand_skills([s.lower().strip() for s in jd.required_skills])
        preferred = self._expand_skills([s.lower().strip() for s in jd.preferred_skills])
        candidate_skills = [s.lower().strip() for s in resume.skills]

        all_jd_skills = required + preferred
        if not all_jd_skills:
            return 50.0, [], []    # no JD skills to match against → neutral score

        # Classify each required skill as matched or missing
        matched = []
        missing = []
        for skill in required:
            if self._skill_present(skill, candidate_skills):
                matched.append(skill)
            else:
                missing.append(skill)

        # Preferred skills contribute to matched list but not missing
        for skill in preferred:
            if self._skill_present(skill, candidate_skills) and skill not in matched:
                matched.append(skill)

        # Compute weighted sub-scores
        required_matched = sum(
            1 for s in required if self._skill_present(s, candidate_skills)
        )
        preferred_matched = sum(
            1 for s in preferred if self._skill_present(s, candidate_skills)
        )

        required_score = (required_matched / len(required) * 100) if required else 100.0
        preferred_score = (preferred_matched / len(preferred) * 100) if preferred else 100.0

        # Required skills count for 70% of keyword score; preferred for 30%
        kw_score = 0.70 * required_score + 0.30 * preferred_score

        return round(kw_score, 2), matched, missing

    @staticmethod
    def _skill_present(skill: str, candidate_skills: List[str]) -> bool:
        """
        Fuzzy skill matching using three strategies:
        1. Exact match:     "python" == "python"
        2. Substring match: "hugging face" in "hugging face transformers"
        3. Acronym match:   "machine learning" → "ml" == "ml"
        """
        skill_lower = skill.lower()
        for cs in candidate_skills:
            cs_lower = cs.lower()
            if skill_lower == cs_lower:
                return True
            if skill_lower in cs_lower or cs_lower in skill_lower:
                return True
            # Build acronym from skill words and compare to candidate
            acronym = "".join(w[0] for w in skill_lower.split() if w)
            if len(acronym) >= 2 and acronym == cs_lower:
                return True
        return False

    # ------------------------------------------------------------------
    # Step 2 – LLM semantic scoring
    # ------------------------------------------------------------------

    def _llm_score(
        self, resume: ParsedResume, jd: ParsedJD
    ) -> Optional[Dict[str, Any]]:
        """
        Call LLM-2 to semantically score *resume* against *jd*.

        Builds a structured prompt using LangChain PromptTemplate,
        sends it to the scorer LLM, and parses the JSON response.

        Returns
        -------
        dict with keys: skills_score, experience_score, education_score,
                        overall_score, explanation
        None if the LLM call or JSON parse fails.
        """
        # Format structured data blocks for the prompt
        jd_block = (
            f"Title: {jd.job_title} | "
            f"Required Skills: {', '.join(jd.required_skills) or 'not specified'} | "
            f"Preferred Skills: {', '.join(jd.preferred_skills) or 'not specified'} | "
            f"Min Experience: {jd.min_experience_years} yrs | "
            f"Education: {jd.education_requirement or 'not specified'} | "
            f"Responsibilities: {'; '.join(jd.responsibilities[:5]) or 'not specified'}"
        )
        resume_block = (
            f"Name: {resume.candidate_name} | "
            f"Skills: {', '.join(resume.skills) or 'none listed'} | "
            f"Experience: {resume.experience_years} yrs – {resume.experience_summary or 'not provided'} | "
            f"Education: {resume.education or 'not provided'} | "
            f"Certifications: {', '.join(resume.certifications) or 'none'}"
        )
        prompt = self._prompt.format(
            candidate_name=resume.candidate_name,
            parsed_jd_data=jd_block,
            structured_resume_data=resume_block,
        )

        try:
            logger.info("LLM scoring '%s' with %s", resume.candidate_name, self.scorer_llm)
            raw = self.scorer_llm.generate(prompt, temperature=0.0)
            data = _extract_json(raw)

            # Validate and clamp pillar scores to their max values
            pillars = data.get("pillar_scores", {})
            pillars["technical"]   = max(0.0, min(40.0, _safe_float(pillars.get("technical",   0))))
            pillars["experience"]  = max(0.0, min(30.0, _safe_float(pillars.get("experience",  0))))
            pillars["soft_skills"] = max(0.0, min(15.0, _safe_float(pillars.get("soft_skills", 0))))
            pillars["impact"]      = max(0.0, min(15.0, _safe_float(pillars.get("impact",      0))))
            data["pillar_scores"]  = pillars
            data["overall_fit_score"] = max(0.0, min(100.0, _safe_float(
                data.get("overall_fit_score",
                         sum(pillars.values()))
            )))
            return data

        except Exception as exc:
            logger.error("LLM scoring failed for '%s': %s", resume.candidate_name, exc)
            return None

    # ------------------------------------------------------------------
    # Fallback heuristics (used when LLM scoring is unavailable)
    # ------------------------------------------------------------------

    @staticmethod
    def _experience_heuristic(resume: ParsedResume, jd: ParsedJD) -> float:
        """
        Rule-based experience score used as fallback when LLM-2 is unavailable.

        Score = min(candidate_years / required_years, 1.5) × 100
        Capped at 100.  Returns 80 if no requirement is stated.
        """
        required = jd.min_experience_years
        candidate = resume.experience_years
        if required <= 0:
            return 80.0    # no requirement specified – generous default
        ratio = min(candidate / required, 1.5)   # cap over-experience benefit
        return round(min(ratio * 100.0, 100.0), 2)
