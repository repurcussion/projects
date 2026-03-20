"""
main.py – Entry point for the CV Sorting Pipeline.

Project Overview
----------------
Automated pipeline that ranks candidate resumes against a job description
using a Dual-LLM architecture orchestrated via LangChain PromptTemplates.

Dual-LLM Design Justification
------------------------------
LLM-1  (gemma:2b / gpt-3.5-turbo) – PARSER
    A lightweight, fast model is used exclusively for structured extraction.
    The task (JSON field extraction from text) is well-defined and does NOT
    require deep reasoning, so a small 2B-parameter model is sufficient and
    avoids wasting expensive GPU/API tokens on a mechanical task.
    LangChain PromptTemplate renders a strict output schema in the prompt
    so the model knows exactly what JSON to return.

LLM-2  (llama3 / gpt-4o) – SCORER
    A larger, reasoning-capable model is used for semantic scoring and
    generating human-readable candidate feedback. This task requires:
      - Nuanced comparison of candidate experience against requirements
      - Understanding of skill synonyms and transferable experience
      - Evidence-based explanation generation
    A 7B+ reasoning model produces significantly better calibrated scores
    and more trustworthy explanations than a 2B parser model would.

LangChain Usage
---------------
- PromptTemplate (parser.py, matcher.py): renders structured prompts with
  named variables; decouples prompt text from Python logic, making prompt
  engineering iterative without code changes.
- All LLM calls go through the BaseLLM.generate() interface, not LangChain
  chains, to keep the code transparent and debuggable for academic review.
- JSON output parsing is done manually (not via LangChain output parsers)
  because smaller Ollama models (gemma:2b) do not reliably follow strict
  Pydantic schemas; our custom _extract_json + _repair_truncated_json
  functions handle the partial/malformed JSON that small models produce.

Pipeline Stages
---------------
1. Input   → file_reader.py  – reads PDF/DOCX/TXT resume files + JD
2. Parse   → parser.py       – LLM-1 extracts structured fields (skills,
                               experience, education) from raw text
3. Match   → matcher.py      – Step A: keyword overlap score (deterministic)
                             – Step B: LLM-2 semantic score (skills 50%,
                               experience 30%, education 20%)
                             – Final: 30% keyword + 70% LLM blend
4. Rank    → ranker.py       – sort by final_score, assign A-F grade
5. Evaluate→ metrics.py      – Precision@K, Recall@K, MRR, NDCG@K
6. Output  → reporter.py     – colour-coded terminal table + results.json

Usage
-----
    python main.py --resumes ./resumes --jd jd.txt
    python main.py --resumes ./resumes --jd jd.txt --demo
    python main.py --resumes ./resumes --jd jd.txt --provider openai --api-key sk-...

Options
-------
    --resumes   PATH    Directory containing resume files (PDF/DOCX/TXT)
    --jd        FILE    Path to the job description text file
    --output    FILE    Output JSON file path (default: results.json)
    --provider  STR     LLM provider: 'ollama' or 'openai' (default: ollama)
    --parser    STR     LLM-1 model name (default: gemma:2b)
    --scorer    STR     LLM-2 model name (default: llama3)
    --topk      INT     K for evaluation metrics (default: 3)
    --api-key   KEY     OpenAI API key (or set OPENAI_API_KEY env var)
    --verbose           Enable verbose debug logging
    --demo              Run with mock LLM responses (no Ollama required)
    --no-report         Skip PDF report and chart generation
    --charts            Generate PNG charts only (no full PDF report)
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Load .env before any config module is imported so env vars are set in time
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=False)
except ImportError:
    pass  # python-dotenv optional; env vars may be set via shell instead

# Prevent Python from writing .pyc files and __pycache__ directories
sys.dont_write_bytecode = True

# ---------------------------------------------------------------------------
# Logging setup – must happen before any module imports that use logging
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s" if verbose else \
          "[%(levelname)s] %(message)s"
    logging.basicConfig(level=level, format=fmt, stream=sys.stdout)
    # Silence noisy third-party loggers unless verbose
    if not verbose:
        for lib in ("httpx", "httpcore", "urllib3", "requests"):
            logging.getLogger(lib).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Module imports (after logging config)
# ---------------------------------------------------------------------------

def _import_modules():
    """
    Deferred import so --help works even without dependencies installed.
    Produces actionable error messages if required packages are missing.
    """
    global FileReader, read_resumes, read_resumes_from_urls, read_job_description
    global ResumeParser, JDParser
    global HybridMatcher
    global Ranker
    global Reporter
    global Evaluator
    global get_parser_llm, get_scorer_llm
    global llm_cfg, scoring_cfg

    try:
        from file_reader import read_resumes, read_resumes_from_urls, read_job_description
        from parser import ResumeParser, JDParser                      # flat import
        from matcher import HybridMatcher                              # flat import
        from ranker import Ranker                                      # flat import
        from reporter import Reporter                                  # flat import
        from metrics import Evaluator                                  # flat import
        from llm_factory import get_parser_llm, get_scorer_llm        # flat import
        from config import llm_cfg, scoring_cfg
    except ImportError as exc:
        # Surface a clear message pointing to requirements.txt
        print(
            f"[ERROR] Missing dependency: {exc}\n"
            "       Install all required packages with:\n"
            "         pip install -r requirements.txt",
            file=sys.stderr,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="CV Sorting using LLMs – ranks candidates against a job description.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --resumes ./resumes --jd jd.txt
  python main.py --resumes ./resumes --jd jd.txt --provider openai --scorer gpt-4o
  python main.py --resumes ./resumes --jd jd.txt --provider hf
  python main.py --resumes ./resumes --jd jd.txt --provider auto
  python main.py --resumes ./resumes --jd jd.txt --demo
  python main.py --resumes ./resumes --jd jd.txt --verbose

URL-based resumes (SAS / presigned URLs, comma-separated):
  python main.py --resumes https://storage.azure.com/cv1.pdf,https://storage.azure.com/cv2.pdf --jd jd.txt
        """,
    )
    p.add_argument(
        "--resumes", required=True, metavar="PATH_OR_URLS",
        help="Local directory containing resume files (PDF/DOCX/DOC/TXT), "
             "or a comma-separated list of HTTPS URLs (SAS/presigned/direct).",
    )
    p.add_argument(
        "--jd", required=True, metavar="FILE",
        help="Path to the job description file (PDF, DOCX, DOC, or TXT)",
    )
    p.add_argument(
        "--output", default="results.json", metavar="FILE",
        help="Output JSON file path (default: results.json)",
    )
    p.add_argument(
        "--provider", default=None, choices=["auto", "ollama", "openai", "hf"],
        help="LLM provider: auto (default: OpenAI→Ollama→HF fallback), "
             "ollama (local only), openai (API only), hf (HuggingFace local only).",
    )
    p.add_argument(
        "--parser", default=None, metavar="MODEL",
        help="LLM-1 model name for parsing (default: gemma:2b)",
    )
    p.add_argument(
        "--scorer", default=None, metavar="MODEL",
        help="LLM-2 model name for scoring (default: llama3)",
    )
    p.add_argument(
        "--topk", type=int, default=None, metavar="K",
        help="K value for Precision@K / Recall@K (default: 3)",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose debug logging",
    )
    p.add_argument(
        "--demo", action="store_true",
        help="Demo mode: uses mock LLM (no Ollama/OpenAI required)",
    )
    p.add_argument(
        "--api-key", default=None, metavar="KEY",
        dest="api_key",
        help="OpenAI API key (alternative to OPENAI_API_KEY env var). Never hardcoded.",
    )
    p.add_argument(
        "--charts", action="store_true",
        help="Generate PNG charts only into output/ (no full PDF report). "
             "Requires matplotlib + numpy.",
    )
    p.add_argument(
        "--no-report", action="store_true", dest="no_report",
        help="Skip automatic PDF report and chart generation.",
    )
    return p


# ---------------------------------------------------------------------------
# Demo / mock LLM (for testing without Ollama)
# ---------------------------------------------------------------------------

class MockLLM:
    """
    Lightweight mock LLM for demonstration purposes.
    Returns plausible but static JSON responses so the full pipeline
    can be tested without a running Ollama instance.
    """
    model_name = "mock-llm"

    # Static parsed resume fields – varied per candidate
    _RESUME_RESPONSES = {
        "alice_chen": {
            "candidate_name": "Alice Chen",
            "email": "alice.chen@email.com",
            "phone": "+1-555-0101",
            "skills": ["Python", "PyTorch", "TensorFlow", "Hugging Face Transformers",
                       "LangChain", "MLflow", "Docker", "Kubernetes", "AWS SageMaker",
                       "FastAPI", "SQL", "OpenAI API", "Apache Spark", "Git"],
            "experience_years": 6,
            "experience_summary": "Senior ML Engineer with 6 years in NLP and production ML systems at DataSphere Inc. and TechVision Labs. Led teams and deployed LLM-based pipelines at scale.",
            "education": "M.Sc. Computer Science (ML), Stanford University",
            "certifications": ["AWS Certified Machine Learning – Specialty", "Deep Learning Specialisation", "LangChain for LLM Application Development"]
        },
        "bob_martinez": {
            "candidate_name": "Bob Martinez",
            "email": "bob.martinez@email.com",
            "phone": "+1-555-0202",
            "skills": ["Python", "SQL", "Apache Spark", "Kafka", "Airflow", "dbt",
                       "Snowflake", "PostgreSQL", "AWS", "Docker", "Pandas", "NumPy"],
            "experience_years": 4,
            "experience_summary": "Data Engineer with 4 years building ETL pipelines and data warehousing. Expertise in Spark, Snowflake, and Airflow. Limited ML experience.",
            "education": "B.Sc. Information Systems, University of Texas",
            "certifications": ["AWS Certified Data Analytics", "Snowflake SnowPro Core"]
        },
        "carol_nguyen": {
            "candidate_name": "Carol Nguyen",
            "email": "carol.nguyen@email.com",
            "phone": "+1-555-0303",
            "skills": ["Python", "PyTorch", "TensorFlow", "JAX", "Hugging Face Transformers",
                       "LangChain", "OpenAI API", "MLflow", "Docker", "FastAPI",
                       "PEFT", "LoRA", "RAG", "FAISS", "Pinecone", "SQL"],
            "experience_years": 5,
            "experience_summary": "ML Research Engineer at MIT with 5 years in NLP and LLMs. Fine-tuned LLaMA 2 and Mistral, built RAG systems, and published 3 papers in top venues.",
            "education": "Ph.D. Candidate Computer Science (NLP), MIT",
            "certifications": ["LangChain Advanced Developer", "TensorFlow Developer Certificate"]
        },
        "david_okafor": {
            "candidate_name": "David Okafor",
            "email": "david.okafor@email.com",
            "phone": "+1-555-0404",
            "skills": ["Python", "Java", "SQL", "Scikit-learn", "Docker",
                       "PostgreSQL", "AWS", "Git", "Flask", "Pandas", "NumPy"],
            "experience_years": 3,
            "experience_summary": "Backend engineer with 3 years in Java/Python microservices. Recently completed ML bootcamp with personal projects in classification and NLP. No production ML experience.",
            "education": "B.Sc. Computer Science, University of Lagos",
            "certifications": ["Machine Learning Specialisation – Coursera", "AWS Cloud Practitioner"]
        },
        "eve_patel": {
            "candidate_name": "Eve Patel",
            "email": "eve.patel@email.com",
            "phone": "+1-555-0505",
            "skills": ["Python", "PyTorch", "TensorFlow", "Hugging Face Transformers",
                       "LangChain", "Docker", "Kubernetes", "MLflow", "AWS SageMaker",
                       "FastAPI", "SQL", "OpenAI API", "OpenCV", "YOLO"],
            "experience_years": 4,
            "experience_summary": "ML Engineer with 4 years in computer vision and NLP at Autonomous Systems Corp and E-Commerce AI. Deployed LangChain document QA and containerised ML services on AWS EKS.",
            "education": "B.Tech. Computer Science and Engineering, IIT Bombay",
            "certifications": ["AWS Certified Machine Learning – Specialty", "Kubernetes Administrator (CKA)", "LangChain Developer Certification"]
        },
    }

    _JD_RESPONSE = {
        "job_title": "Senior Machine Learning Engineer",
        "required_skills": ["Python", "PyTorch", "TensorFlow", "Hugging Face Transformers",
                            "LangChain", "MLflow", "Docker", "Kubernetes", "AWS", "SQL",
                            "FastAPI", "Git"],
        "preferred_skills": ["LLMs", "RAG", "OpenAI API", "Apache Spark", "MLOps",
                             "LoRA", "PEFT", "Pinecone", "FAISS"],
        "min_experience_years": 4,
        "education_requirement": "Bachelor's in Computer Science or related field",
        "responsibilities": [
            "Design and implement scalable ML pipelines for NLP tasks",
            "Fine-tune and deploy large language models",
            "Build LangChain-based applications including RAG systems",
            "Establish MLOps best practices",
            "Mentor junior engineers",
        ]
    }

    _SCORE_RESPONSES = {
        "Alice Chen": {
            "skills_score": 92, "experience_score": 95, "education_score": 100,
            "overall_score": 94,
            "explanation": "Alice has all 12 required skills and 7 preferred skills. Her 6 years of experience significantly exceeds the 4-year requirement, with direct LLM production experience. M.Sc. from Stanford exceeds the education requirement. Minor gap: no explicit RAG project mentioned but LLM deployment work is highly relevant."
        },
        "Bob Martinez": {
            "skills_score": 28, "experience_score": 40, "education_score": 60,
            "overall_score": 35,
            "explanation": "Bob has Python and SQL but is missing 8 required skills including PyTorch, TensorFlow, Hugging Face Transformers, LangChain, Kubernetes, and MLflow. His 4 years experience meets the minimum but is in data engineering, not ML engineering. Education meets minimum requirement."
        },
        "Carol Nguyen": {
            "skills_score": 96, "experience_score": 85, "education_score": 100,
            "overall_score": 93,
            "explanation": "Carol has all required skills plus extensive preferred skills including RAG, LoRA, PEFT, Pinecone, and FAISS. Her 5 years in ML research with LLM fine-tuning is highly relevant. Ph.D. (ABD) from MIT exceeds all education requirements. Small gap: primarily research background with less production deployment experience versus Alice."
        },
        "David Okafor": {
            "skills_score": 18, "experience_score": 25, "education_score": 60,
            "overall_score": 22,
            "explanation": "David is missing most required ML skills: no PyTorch/TensorFlow production experience, no LangChain, no Kubernetes, no Hugging Face Transformers in production. His 3 years are in backend engineering, not ML. Personal ML projects show interest but do not substitute for professional experience. Not recommended for a senior role."
        },
        "Eve Patel": {
            "skills_score": 88, "experience_score": 80, "education_score": 80,
            "overall_score": 85,
            "explanation": "Eve has 12 required skills and multiple preferred skills including LangChain, OpenAI API, and MLflow. Her 4 years in ML engineering meets the minimum with strong production deployments. B.Tech. from IIT Bombay meets but does not exceed education requirement. Primarily computer vision background but LangChain QA system demonstrates NLP capability."
        },
    }

    def generate(self, prompt: str, temperature: float = 0.1) -> str:
        import json, time
        time.sleep(0.1)  # Simulate small latency

        # Detect which type of prompt it is
        if '"candidate_name"' in prompt or "Resume Text" in prompt:
            # Resume parsing prompt – find which candidate
            for name_key, data in self._RESUME_RESPONSES.items():
                if name_key.replace("_", " ").split()[0].lower() in prompt.lower():
                    return json.dumps(data)
            # Default
            return json.dumps(self._RESUME_RESPONSES["alice_chen"])

        elif "required_skills" in prompt and "job_title" in prompt and "responsibilities" in prompt \
             and "CANDIDATE PROFILE" not in prompt:
            # JD parsing prompt
            return json.dumps(self._JD_RESPONSE)

        elif "skills_score" in prompt or "SCORING RULES" in prompt:
            # Scoring prompt – find which candidate
            for candidate_name, scores in self._SCORE_RESPONSES.items():
                if candidate_name.split()[0] in prompt:
                    return json.dumps(scores)
            return json.dumps(self._SCORE_RESPONSES["Alice Chen"])

        # Generic fallback
        return json.dumps({"result": "ok"})


# ---------------------------------------------------------------------------
# Validation and initialisation helper
# ---------------------------------------------------------------------------

def _validate_and_init(args: argparse.Namespace, logger: logging.Logger):
    """
    Validate CLI inputs, apply config overrides, and initialise LLMs.

    Separated from run_pipeline so that path/config errors are caught before
    any expensive file I/O or model loading begins.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments from _build_arg_parser().
    logger : logging.Logger
        Logger instance from the calling pipeline function.

    Returns
    -------
    tuple (parser_llm, scorer_llm)
        Ready-to-use LLM instances on success.
    None
        On any validation or initialisation failure (error already logged).
    """
    # ------------------------------------------------------------------
    # 1. Early path validation – fail fast before any LLM calls
    # ------------------------------------------------------------------
    _resumes_are_urls = args.resumes.startswith(("http://", "https://"))
    resumes_path = Path(args.resumes) if not _resumes_are_urls else None
    jd_path = Path(args.jd)

    if not _resumes_are_urls and (
        resumes_path is None
        or not resumes_path.exists()
        or not resumes_path.is_dir()
    ):
        logger.error(
            "Resumes directory not found: '%s'\n"
            "  Create the directory and add resume files (PDF/DOCX/TXT) inside,\n"
            "  or pass comma-separated HTTPS URLs instead.",
            args.resumes,
        )
        return None

    if not jd_path.exists() or not jd_path.is_file():
        logger.error(
            "Job description file not found: '%s'\n"
            "  Provide a valid path with --jd <file>.",
            args.jd,
        )
        return None

    # ------------------------------------------------------------------
    # 2. Apply CLI overrides to config (env vars + dataclass fields)
    # ------------------------------------------------------------------
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider
        llm_cfg.provider = args.provider
    if args.parser:
        os.environ["PARSER_MODEL"] = args.parser
        llm_cfg.parser_model = args.parser
    if args.scorer:
        os.environ["SCORER_MODEL"] = args.scorer
        llm_cfg.scorer_model = args.scorer
    if args.topk:
        if args.topk < 1:
            logger.error("--topk must be a positive integer (got %d).", args.topk)
            return None
        scoring_cfg.top_k = args.topk
    # API key: CLI flag takes precedence over OPENAI_API_KEY env var
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
        llm_cfg.openai_api_key = args.api_key
        logger.info("OpenAI API key provided via --api-key flag.")

    # ------------------------------------------------------------------
    # 3. LLM initialisation
    # ------------------------------------------------------------------
    if args.demo:
        # Demo mode: return a shared MockLLM instance for both roles
        logger.info("DEMO MODE: Using mock LLM (no Ollama required)")
        mock = MockLLM()
        return mock, mock

    logger.info("Initialising LLMs (provider: %s)", llm_cfg.provider)
    try:
        parser_llm = get_parser_llm()
        scorer_llm = get_scorer_llm()
        return parser_llm, scorer_llm
    except EnvironmentError as exc:
        # Missing API key for OpenAI provider
        logger.error("%s", exc)
        logger.error(
            "Set the key with: export OPENAI_API_KEY='sk-...' "
            "or pass --api-key <key>"
        )
        return None
    except Exception as exc:
        err_str = str(exc).lower()
        if "connection" in err_str or "refused" in err_str or "timeout" in err_str:
            # Ollama server is not running
            logger.error(
                "Cannot connect to Ollama at %s\n"
                "  Start the server with: ollama serve\n"
                "  Or use demo mode:       python main.py --demo ...",
                llm_cfg.ollama_base_url,
            )
        else:
            logger.error("Failed to initialise LLMs: %s", exc)
            logger.error("Tip: Use --demo flag to run without Ollama/OpenAI.")
        return None


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> int:
    """
    Execute the full CV sorting pipeline and return exit code.

    Pipeline stages
    ---------------
    1. Input   → Read resumes + JD from files
    2. Parse   → LLM-1 extracts structured fields
    3. Match   → Keyword + LLM-2 hybrid scoring
    4. Rank    → Sort by final score
    5. Output  → Terminal display + results.json
    6. Evaluate→ Compute metrics
    """
    _import_modules()

    logger = logging.getLogger("pipeline")
    total_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Step 0: Validate paths, apply config overrides, initialise LLMs
    # ------------------------------------------------------------------
    init_result = _validate_and_init(args, logger)
    if init_result is None:
        return 1   # error already logged inside _validate_and_init
    parser_llm, scorer_llm = init_result

    # ------------------------------------------------------------------
    # Step 1: Input – read files
    # ------------------------------------------------------------------
    logger.info("Reading resumes from: %s", args.resumes)
    try:
        from file_reader import read_resumes, read_resumes_from_urls, read_job_description
        if args.resumes.startswith(("http://", "https://")):
            urls = [u.strip() for u in args.resumes.split(",") if u.strip()]
            raw_resumes = read_resumes_from_urls(urls)
        else:
            raw_resumes = read_resumes(args.resumes)
        raw_jd = read_job_description(args.jd)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Input error: %s", exc)
        return 1

    logger.info("Loaded %d resumes | JD: %d chars", len(raw_resumes), len(raw_jd))

    # ------------------------------------------------------------------
    # Step 2: Parse – LLM-1 extracts structured data
    # ------------------------------------------------------------------
    parse_start = time.perf_counter()

    from parser import ResumeParser, JDParser
    resume_parser = ResumeParser(llm=parser_llm)
    jd_parser = JDParser(llm=parser_llm)

    logger.info("Parsing job description...")
    parsed_jd = jd_parser.parse(raw_jd)

    logger.info("Parsing %d resumes...", len(raw_resumes))
    parsed_resumes = resume_parser.parse_batch(raw_resumes)

    parse_time = time.perf_counter() - parse_start
    logger.info("Parsing complete in %.2fs", parse_time)

    # Filter out completely failed parses but keep partial ones
    # A resume is considered usable if it has at least skills OR a name
    valid_resumes = [r for r in parsed_resumes if not (
        r.parse_error and not r.skills and not r.experience_summary
    )]
    skipped = len(parsed_resumes) - len(valid_resumes)
    if skipped:
        logger.warning(
            "%d resume(s) had unrecoverable parse errors and were skipped.",
            skipped,
        )

    if not valid_resumes:
        logger.error(
            "No resumes could be parsed.\n"
            "  Check that resume files contain readable text and retry."
        )
        return 1

    # Clamp top_k to the actual number of candidates to avoid misleading metrics
    if scoring_cfg.top_k > len(valid_resumes):
        logger.warning(
            "--topk (%d) exceeds candidate count (%d); clamping to %d.",
            scoring_cfg.top_k, len(valid_resumes), len(valid_resumes),
        )
        scoring_cfg.top_k = len(valid_resumes)

    # ------------------------------------------------------------------
    # Step 3: Match – keyword + LLM-2 hybrid scoring
    # ------------------------------------------------------------------
    match_start = time.perf_counter()

    from matcher import HybridMatcher
    matcher = HybridMatcher(scorer_llm=scorer_llm)

    logger.info("Matching %d candidates against JD...", len(valid_resumes))
    match_results = matcher.match_batch(valid_resumes, parsed_jd)

    match_time = time.perf_counter() - match_start
    logger.info("Matching complete in %.2fs", match_time)

    # ------------------------------------------------------------------
    # Step 4: Rank – sort by final score
    # ------------------------------------------------------------------
    from ranker import Ranker
    ranker = Ranker()
    ranked = ranker.rank(match_results)

    # ------------------------------------------------------------------
    # Step 5: Evaluate – compute metrics
    # ------------------------------------------------------------------
    total_time = time.perf_counter() - total_start

    from metrics import Evaluator
    evaluator = Evaluator(top_k=scoring_cfg.top_k)
    metrics = evaluator.compute_metrics(
        ranked=ranked,
        parse_time=parse_time,
        match_time=match_time,
        total_time=total_time,
        parser_model=llm_cfg.parser_model,
        scorer_model=llm_cfg.scorer_model,
        provider=llm_cfg.provider,
    )

    # ------------------------------------------------------------------
    # Step 6: Output – display + save
    # ------------------------------------------------------------------
    from reporter import Reporter
    reporter = Reporter(output_path=args.output)

    reporter.print_header(parsed_jd, len(ranked))
    reporter.print_ranked_results(ranked)
    reporter.print_evaluation_metrics(metrics)

    # Save results to JSON – gracefully handle write failures
    try:
        saved_path = reporter.save_results(
            ranked=ranked,
            jd=parsed_jd,
            metrics=metrics,
            parsed_resumes=parsed_resumes,
        )
        reporter.print_footer(saved_path)
    except (OSError, PermissionError) as exc:
        # Write failed (bad path, no disk space, no permissions)
        logger.error(
            "Could not write results to '%s': %s\n"
            "  Check the output path and available disk space.",
            args.output, exc,
        )
        # Degrade gracefully: print top result to stdout so work is not lost
        logger.info("Top candidate: %s (%.1f/100)", ranked[0].candidate_name, ranked[0].final_score)

    # ------------------------------------------------------------------
    # Step 7: Chart generation (always runs unless --no-report)
    # Charts are intermediate files – always written to output/
    # ------------------------------------------------------------------
    # Charts run in default mode OR when --charts flag is explicitly set
    charts_dir = None
    if not args.no_report or args.charts:
        charts_dir = reporter.save_charts(
            ranked=ranked,
            metrics=metrics,
            output_dir="output",   # intermediate files always go to output/
        )

    # ------------------------------------------------------------------
    # Step 8: PDF report generation (default, skip with --no-report)
    # Report.pdf is the only file written to Report/
    # ------------------------------------------------------------------
    if not args.no_report and not args.charts:
        try:
            from report_generator import ReportGenerator
            rg = ReportGenerator(output_dir="Report", charts_dir="output")
            report_path = rg.generate(ranked=ranked, jd=parsed_jd, metrics=metrics)
            if report_path:
                logger.info("Report saved to '%s'", report_path)
                print(f"Report saved to: {report_path}")
        except ImportError:
            logger.warning(
                "reportlab not installed \u2013 skipping PDF report. "
                "Install with: pip install reportlab"
            )

    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_arg_parser()

    # Show full help when called with no arguments instead of a terse error
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    _setup_logging(args.verbose)
    logger = logging.getLogger("main")

    logger.info("CV Sorting Pipeline starting...")
    logger.info("Resumes: %s | JD: %s", args.resumes, args.jd)
    if args.demo:
        logger.info("Running in DEMO mode (mock LLM)")

    exit_code = run_pipeline(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
