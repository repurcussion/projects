"""
parser.py – Structured extraction layer using LLM-1 (gemma:2b / gpt-3.5-turbo).

Uses LangChain PromptTemplate to build prompts that instruct the LLM to
return structured JSON from raw resume or job-description text.

Robustness layers (in order of preference):
1. LLM returns valid JSON   → parse directly
2. LLM returns truncated JSON → _repair_truncated_json() closes open brackets
3. LLM returns unusable output → _regex_resume_fallback() scans known tech skills
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# LangChain v1.x moved prompts to langchain_core; fall back for older versions
try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    from langchain.prompts import PromptTemplate

from llm_base import BaseLLM   # flat import

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured output dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ParsedResume:
    """
    Structured representation of a parsed candidate resume.

    Fields populated by LLM-1 (or regex fallback) from raw resume text.
    """
    candidate_name: str = "Unknown"
    email: str = ""
    phone: str = ""
    skills: List[str] = field(default_factory=list)       # technical + soft skills
    experience_years: float = 0.0                          # total professional years
    experience_summary: str = ""                           # 1-sentence summary
    education: str = ""                                    # highest qualification
    certifications: List[str] = field(default_factory=list)
    raw_text: str = ""                                     # original resume text
    parse_error: Optional[str] = None                     # set if LLM parse failed

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict for JSON output."""
        return {
            "candidate_name": self.candidate_name,
            "email": self.email,
            "phone": self.phone,
            "skills": self.skills,
            "experience_years": self.experience_years,
            "experience_summary": self.experience_summary,
            "education": self.education,
            "certifications": self.certifications,
            "parse_error": self.parse_error,
        }


@dataclass
class ParsedJD:
    """
    Structured representation of a parsed job description.

    Fields populated by LLM-1 (or regex fallback) from raw JD text.
    """
    job_title: str = "Unknown Role"
    required_skills: List[str] = field(default_factory=list)   # must-have skills
    preferred_skills: List[str] = field(default_factory=list)  # nice-to-have skills
    min_experience_years: float = 0.0
    education_requirement: str = ""
    responsibilities: List[str] = field(default_factory=list)
    raw_text: str = ""
    parse_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict for JSON output."""
        return {
            "job_title": self.job_title,
            "required_skills": self.required_skills,
            "preferred_skills": self.preferred_skills,
            "min_experience_years": self.min_experience_years,
            "education_requirement": self.education_requirement,
            "responsibilities": self.responsibilities,
            "parse_error": self.parse_error,
        }


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

# Resume parsing prompt – designed for gemma:2b's limited output window.
# Asks for max 12 specific tool names to keep the JSON response short.
RESUME_PARSE_TEMPLATE = """\
You are a resume parser. Extract information from the resume below.

STRICT RULES:
- skills: list up to 12 specific technology/tool names ONLY (e.g. "Python", "Docker", "PyTorch").
  Do NOT include category labels like "Core ML:", "Programming:", "NLP:".
  Do NOT put multiple skills in one string. Each skill must be a single short name.
- experience_years: total years of work experience as a number. Use 0 if unknown.
- experience_summary: ONE sentence only summarising job history.
- education: highest degree and institution only (e.g. "MSc Computer Science, MIT").
- If a field is missing use "" or [].

Resume:
{resume_text}

Reply with ONLY this JSON (no markdown, no extra text, no trailing commas):
{{"candidate_name":"","email":"","phone":"","skills":[],"experience_years":0,"experience_summary":"","education":""}}
"""

# JD parsing prompt – extracts required/preferred skills and role details.
JD_PARSE_TEMPLATE = """\
You are a job description analyst. Extract structured requirements from the job description below.

RULES:
- required_skills: skills explicitly marked as required or must-have.
- preferred_skills: skills marked as nice-to-have or preferred.
- min_experience_years: minimum years of experience as a number. If not stated, use 0.
- education_requirement: minimum education level required.
- responsibilities: list the key job responsibilities (max 6).
- Do NOT infer or add skills not mentioned in the text.

Job Description:
-----------------
{jd_text}
-----------------

Respond ONLY with valid JSON matching this exact schema (no markdown, no extra text):
{{
  "job_title": "string",
  "required_skills": ["string", ...],
  "preferred_skills": ["string", ...],
  "min_experience_years": number,
  "education_requirement": "string",
  "responsibilities": ["string", ...]
}}
"""


# ---------------------------------------------------------------------------
# JSON extraction with partial-JSON repair
# ---------------------------------------------------------------------------

def _repair_truncated_json(text: str) -> str:
    """
    Attempt to close a JSON object that was cut off mid-stream by the LLM
    hitting its token limit.

    Strategy:
    1. Drop any trailing incomplete token (partial word / number).
    2. Walk the string tracking open/close brackets and string state.
    3. Close any open string, then close all unbalanced brackets in reverse.
    """
    s = text.rstrip()

    # Drop trailing incomplete token: anything that isn't a valid JSON terminal.
    # Include the last safe character (last_safe+1) so an opening `"` is
    # preserved – the walk below then tracks it as an open string and closes it.
    if s and s[-1] not in ('"', ']', '}', 'e', 'l', 'n',
                            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'):
        last_safe = max(s.rfind('"'), s.rfind(','), s.rfind('['), s.rfind('{'))
        if last_safe > 0:
            s = s[:last_safe + 1]   # include the safe char itself

    # Walk string to find unbalanced brackets
    open_stack: list = []
    in_string = False
    i = 0
    while i < len(s):
        c = s[i]
        if c == '\\' and in_string:
            i += 2          # skip escaped character
            continue
        if c == '"':
            in_string = not in_string
        elif not in_string:
            if c in ('{', '['):
                open_stack.append(c)
            elif c == '}' and open_stack and open_stack[-1] == '{':
                open_stack.pop()
            elif c == ']' and open_stack and open_stack[-1] == '[':
                open_stack.pop()
        i += 1

    # Close any unterminated string literal
    if in_string:
        s += '"'

    # Strip trailing comma before closing (JSON forbids trailing commas)
    s = s.rstrip().rstrip(',')

    # Close all remaining open brackets in reverse order
    for b in reversed(open_stack):
        s += ']' if b == '[' else '}'

    return s


def _extract_json(text: str) -> Dict[str, Any]:
    """
    Robustly extract the first valid JSON object from *text*.

    Attempt order:
    1. Direct json.loads()                       – clean model output
    2. Strip markdown fences, retry              – model wrapped in ```json
    3. Extract first {...} block, retry          – leading/trailing text
    4. Repair truncated JSON, retry              – token-limit cutoff
    """
    # Attempt 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Attempt 2: strip markdown code fences
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip().strip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Attempt 3: find the first {...} block (ignore any surrounding prose)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    candidate = match.group(0) if match else cleaned
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Attempt 4: sanitise literal control characters (gemma:2b often embeds
    # raw newlines inside JSON string values, which makes the output invalid)
    sanitised = re.sub(r'[\r\n\t]+', ' ', candidate)
    try:
        return json.loads(sanitised)
    except json.JSONDecodeError:
        pass

    # Attempt 5: repair truncated JSON, then sanitise
    repaired = _repair_truncated_json(sanitised)
    try:
        data = json.loads(repaired)
        logger.warning("Partial JSON repaired and recovered.")
        return data
    except json.JSONDecodeError:
        pass

    raise ValueError(f"No valid JSON found in LLM output:\n{text[:300]}")


# ---------------------------------------------------------------------------
# Skill post-processing helpers
# ---------------------------------------------------------------------------

def _clean_skills(skills: List[str]) -> List[str]:
    """
    Post-process LLM skill output to fix two common gemma:2b errors:

    1. Strip category prefixes: "Core ML: Python, PyTorch" → ["Python", "PyTorch"]
    2. Split comma-merged strings: "Python, Docker" → ["Python", "Docker"]
    3. Deduplicate while preserving insertion order.
    """
    cleaned: List[str] = []
    for item in skills:
        # Remove leading "Category:" or "Category –" labels (colon or en-dash only).
        # Do NOT strip hyphens: they appear in skill names like Scikit-learn, XGBoost.
        item = re.sub(r'^[A-Za-z /&]+[:\u2013]\s*', '', item).strip()
        # Split if multiple skills were merged into one element
        parts = [p.strip() for p in item.split(',')] if ',' in item else [item]
        cleaned.extend(p for p in parts if p)

    # Deduplicate while preserving order
    seen: set = set()
    result: List[str] = []
    for s in cleaned:
        key = s.lower()
        if key not in seen and len(s) > 1:
            seen.add(key)
            result.append(s)
    return result


# Master list of known ML/tech skill names for deterministic regex fallback
_KNOWN_TECH_SKILLS = [
    "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Go", "Rust", "SQL", "Bash", "R",
    "PyTorch", "TensorFlow", "Keras", "JAX", "Scikit-learn", "XGBoost", "LightGBM", "CatBoost",
    "Hugging Face", "Transformers", "LangChain", "LangGraph", "LlamaIndex", "OpenAI API",
    "LLaMA", "Mistral", "Gemma", "GPT", "BERT", "T5", "PEFT", "LoRA", "RAG", "RLHF",
    "spaCy", "NLTK", "FastAPI", "Flask", "Django", "REST", "GraphQL",
    "Docker", "Kubernetes", "Helm", "Terraform", "Ansible", "CI/CD", "GitHub Actions",
    "AWS", "GCP", "Azure", "SageMaker", "Vertex AI", "Lambda", "S3", "EC2",
    "MLflow", "DVC", "Weights & Biases", "W&B", "Comet", "Airflow",
    "Spark", "Kafka", "Hadoop", "Flink", "Hive", "Snowflake", "Redshift", "BigQuery",
    "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch", "Pinecone", "FAISS", "ChromaDB",
    "Pandas", "NumPy", "SciPy", "Matplotlib", "Seaborn", "Plotly",
    "Git", "Linux", "OpenCV", "YOLO", "Stable Diffusion",
]


def _regex_extract_skills(text: str) -> List[str]:
    """
    Scan *text* for known technology/tool names and return those found.

    Used as a deterministic fallback when the LLM returns unusable output.
    Matches case-insensitively against _KNOWN_TECH_SKILLS.
    """
    found: List[str] = []
    text_lower = text.lower()
    for skill in _KNOWN_TECH_SKILLS:
        if skill.lower() in text_lower:
            found.append(skill)
    return found


def _regex_resume_fallback(
    candidate_name: str, raw_text: str, error: Optional[str]
) -> "ParsedResume":
    """
    Build a ParsedResume entirely from regex when the LLM fails completely.

    Extracts: name (from first line), email, phone, years of experience,
    experience summary (from SUMMARY section), education, and tech skills.
    This guarantees every candidate enters the matching stage with a skills list.
    """
    # Extract name: use the first non-empty line if it looks like a person's name
    lines = [l.strip() for l in raw_text.strip().splitlines() if l.strip()]
    if lines:
        first = lines[0]
        if re.match(r'^[A-Za-z][A-Za-z\s\-\.]+$', first) and len(first.split()) <= 4:
            candidate_name = first

    # Extract email via standard pattern
    email_m = re.search(r"[\w.+-]+@[\w.-]+\.\w+", raw_text)
    # Extract phone (7-15 digit sequence with common separators)
    phone_m = re.search(r"\+?[\d\s\-().]{7,15}", raw_text)
    # Extract years of experience mentioned in text
    years_m = re.search(r"(\d+(?:\.\d+)?)\s*\+?\s*years?", raw_text, re.IGNORECASE)
    # Extract education level and institution
    edu_m = re.search(
        r"(B\.?S\.?c?|M\.?S\.?c?|M\.?Eng|Ph\.?D|Bachelor|Master|Doctor)[^\n]{0,60}",
        raw_text, re.IGNORECASE,
    )
    # Extract first sentence of SUMMARY section for a meaningful experience summary
    summary_text = ""
    summary_m = re.search(r'SUMMARY\s*\n(.*?)(?:\n\n|\Z)', raw_text,
                           re.IGNORECASE | re.DOTALL)
    if summary_m:
        first_sentence = re.split(r'(?<=[.!?])\s+', summary_m.group(1).strip())
        summary_text = first_sentence[0] if first_sentence else ""
    if not summary_text and years_m:
        summary_text = f"{int(_safe_float(years_m.group(1)))} years of professional experience."

    # Scan raw text for known tech skill keywords
    skills = _regex_extract_skills(raw_text)

    return ParsedResume(
        candidate_name=candidate_name,
        email=email_m.group(0) if email_m else "",
        phone=phone_m.group(0).strip() if phone_m else "",
        skills=skills,
        experience_years=_safe_float(years_m.group(1)) if years_m else 0.0,
        experience_summary=summary_text or "Extracted via regex fallback.",
        education=edu_m.group(0).strip() if edu_m else "",
        raw_text=raw_text,
        parse_error=error,
    )


# ---------------------------------------------------------------------------
# ResumeParser
# ---------------------------------------------------------------------------

class ResumeParser:
    """
    Parses raw resume text into a ParsedResume dataclass using LLM-1.

    Processing order:
    1. LLM call → _extract_json() (includes partial JSON repair)
    2. _clean_skills() to fix category-prefixed or comma-merged skill lists
    3. If LLM skills < 4, supplement/replace with _regex_extract_skills()
    4. Full regex fallback if LLM call fails entirely

    Parameters
    ----------
    llm : BaseLLM
        The parser LLM instance (e.g. OllamaLLM("gemma:2b")).
    """

    def __init__(self, llm: BaseLLM):
        self.llm = llm
        # LangChain PromptTemplate renders the prompt with {resume_text} filled in
        self._prompt = PromptTemplate(
            input_variables=["resume_text"],
            template=RESUME_PARSE_TEMPLATE,
        )

    def parse(self, candidate_name: str, raw_text: str) -> ParsedResume:
        """
        Extract structured fields from *raw_text* for *candidate_name*.

        Returns a ParsedResume on success.
        On LLM failure, falls back to regex extraction so the pipeline
        always has at least skills and basic metadata for matching.
        """
        if not raw_text.strip():
            logger.warning("Empty resume text for '%s'.", candidate_name)
            return ParsedResume(
                candidate_name=candidate_name,
                raw_text=raw_text,
                parse_error="Empty resume text",
            )

        # Truncate to 2000 chars – keeps gemma:2b well within its output budget
        # while preserving the most important skills/experience at the top
        truncated = raw_text[:2000]
        prompt = self._prompt.format(resume_text=truncated)

        llm_data: Optional[Dict[str, Any]] = None
        parse_error: Optional[str] = None
        try:
            logger.info("Parsing resume for '%s' with %s", candidate_name, self.llm)
            raw_output = self.llm.generate(prompt, temperature=0.0)
            llm_data = _extract_json(raw_output)
        except Exception as exc:
            parse_error = str(exc)
            logger.warning(
                "LLM parse failed for '%s', using regex fallback: %s",
                candidate_name, exc,
            )

        if llm_data is not None:
            # Post-process skills: strip category labels, split merged lists
            raw_skills = _to_str_list(llm_data.get("skills", []))
            skills = _clean_skills(raw_skills)
            # Supplement with regex when LLM extracted fewer than 8 skills, OR
            # when regex finds 10+ more skills than LLM (partial extraction: Eve
            # gets 16 from LLM but 28+ from regex due to token budget pressure).
            regex_skills = _regex_extract_skills(raw_text)
            if len(skills) < 8 or len(regex_skills) > len(skills) + 10:
                existing_lower = {s.lower() for s in skills}
                for s in regex_skills:
                    if s.lower() not in existing_lower:
                        skills.append(s)
                        existing_lower.add(s.lower())

            # If LLM set experience_years to 0, try to recover from summary/text
            exp_years = _safe_float(llm_data.get("experience_years", 0))
            if exp_years == 0:
                summary_text = str(llm_data.get("experience_summary", ""))
                years_m = re.search(
                    r"(\d+(?:\.\d+)?)\s*\+?\s*years?",
                    summary_text + " " + raw_text[:500],
                    re.IGNORECASE,
                )
                if years_m:
                    exp_years = _safe_float(years_m.group(1))

            return ParsedResume(
                candidate_name=llm_data.get("candidate_name") or candidate_name,
                email=str(llm_data.get("email", "")),
                phone=str(llm_data.get("phone", "")),
                skills=skills,
                experience_years=exp_years,
                experience_summary=str(llm_data.get("experience_summary", "")),
                education=str(llm_data.get("education", "")),
                certifications=_to_str_list(llm_data.get("certifications", [])),
                raw_text=raw_text,
            )

        # Full regex fallback: LLM output was completely unrecoverable
        logger.warning("Using full regex fallback for '%s'.", candidate_name)
        return _regex_resume_fallback(candidate_name, raw_text, parse_error)

    def parse_batch(self, resumes: Dict[str, str]) -> List[ParsedResume]:
        """
        Parse multiple resumes sequentially.

        Parameters
        ----------
        resumes : dict
            {candidate_name: raw_text} mapping from FileReader.

        Returns
        -------
        list of ParsedResume
            One entry per resume, including partial/fallback results.
        """
        results = []
        for name, text in resumes.items():
            results.append(self.parse(name, text))
        return results


# ---------------------------------------------------------------------------
# JDParser
# ---------------------------------------------------------------------------

class JDParser:
    """
    Parses raw job description text into a ParsedJD dataclass using LLM-1.

    Falls back to simple keyword extraction if the LLM fails.

    Parameters
    ----------
    llm : BaseLLM
        The parser LLM instance (same as resume parser).
    """

    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self._prompt = PromptTemplate(
            input_variables=["jd_text"],
            template=JD_PARSE_TEMPLATE,
        )

    def parse(self, raw_text: str) -> ParsedJD:
        """
        Extract structured requirements from *raw_text*.

        Returns ParsedJD on success.  On failure, uses _fallback_jd_parse()
        to extract capitalised skill tokens as a best-effort alternative.
        """
        if not raw_text.strip():
            raise ValueError("Job description text is empty.")

        # Truncate to 3000 chars – JD is usually shorter than resumes
        truncated = raw_text[:3000]
        prompt = self._prompt.format(jd_text=truncated)

        try:
            logger.info("Parsing job description with %s", self.llm)
            raw_output = self.llm.generate(prompt, temperature=0.0)
            data = _extract_json(raw_output)
        except Exception as exc:
            logger.error("JD parse failed: %s – using fallback.", exc)
            return _fallback_jd_parse(raw_text, str(exc))

        return ParsedJD(
            job_title=str(data.get("job_title", "Unknown Role")),
            required_skills=_to_str_list(data.get("required_skills", [])),
            preferred_skills=_to_str_list(data.get("preferred_skills", [])),
            min_experience_years=_safe_float(data.get("min_experience_years", 0)),
            education_requirement=str(data.get("education_requirement", "")),
            responsibilities=_to_str_list(data.get("responsibilities", [])),
            raw_text=raw_text,
        )


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _to_str_list(value: Any) -> List[str]:
    """
    Coerce an LLM output field to a list of non-empty strings.

    Handles: actual list, single string, comma-separated string.
    """
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        # Handle case where LLM returned a comma-separated string instead of a list
        if ',' in value:
            return [p.strip() for p in value.split(',') if p.strip()]
        return [value.strip()]
    return []


def _safe_float(value: Any) -> float:
    """Convert *value* to float, returning 0.0 on any failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _fallback_jd_parse(raw_text: str, error: str) -> ParsedJD:
    """
    Simple regex-based fallback when the LLM fails to parse a JD.

    Extracts capitalised skill-like tokens (e.g. "Python", "Docker", "AWS")
    as a best-effort required_skills list.
    """
    logger.warning("Using fallback JD parsing (keyword extraction).")
    # Match capitalised words and common tech abbreviations
    words = re.findall(r"\b[A-Z][a-z]+(?:\+\+|#|\.js|\.py)?\b|\b[A-Z]{2,}\b", raw_text)
    skills = list(dict.fromkeys(words))[:20]   # deduplicate, preserve order, cap at 20
    return ParsedJD(
        job_title="Unknown Role",
        required_skills=skills,
        raw_text=raw_text,
        parse_error=error,
    )
