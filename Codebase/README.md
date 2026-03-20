# SmartRank: A Multi-LLM Framework for Automated CV Sorting & Evaluation

A production-quality, modular Python pipeline that uses a cascading multi-LLM architecture (OpenAI API → Ollama → HuggingFace) to parse resumes and job descriptions, perform hybrid candidate-job matching, and output a ranked leaderboard with explanations and evaluation metrics.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        SMARTRANK                             │
├──────────────┬──────────────┬──────────────┬────────────────┤
│  INPUT LAYER │ PARSING LAYER│  LLM LAYER   │ MATCHING ENGINE│
│              │              │              │                │
│ file_reader  │  parser.py   │  base.py     │  matcher.py    │
│ .pdf/.docx   │  LLM-1 →     │  ollama_llm  │  Step1: Keyword│
│ .txt files   │  ParsedResume│  openai_llm  │  Step2: LLM-2  │
│              │  ParsedJD    │  factory.py  │  Hybrid blend  │
└──────────────┴──────────────┴──────────────┴────────────────┘
       ↓               ↓              ↓              ↓
┌──────────────┬──────────────┬──────────────┬────────────────┐
│RANKING ENGINE│ OUTPUT LAYER │  EVALUATION  │   CONFIG       │
│              │              │              │                │
│  ranker.py   │  reporter.py │  metrics.py  │  config.py     │
│  Sort by     │  Terminal    │  Precision@K │  Weights       │
│  score       │  display     │  Recall@K    │  Models        │
│  Grade A-F   │  results.json│  MRR, NDCG   │  Thresholds    │
└──────────────┴──────────────┴──────────────┴────────────────┘
```

---

## LLM Design

| Role | Model (Ollama) | Model (OpenAI) | Model (HuggingFace) | Purpose |
|------|---------------|----------------|---------------------|--------|
| **LLM-1** (Parser) | `gemma:2b` | `gpt-3.5-turbo` | `flan-t5-base` | Fast extraction – structured JSON from raw resume + JD text |
| **LLM-2** (Scorer) | `llama3` | `gpt-4o` | — | Reasoning – semantic scoring + human-readable explanation |

> **Default provider is `auto`**: tries OpenAI API first, falls back to Ollama, then HuggingFace.

Both models are injected via the `BaseLLM` interface. Swapping models requires a single config change.

---

## Quick Start

### 1 – Install dependencies
```bash
pip install -r requirements.txt
```

### 2a – Demo mode (no Ollama required)
```bash
python main.py --resumes ./resumes --jd jd.txt --demo
```

### 2b – Auto mode (API first, local fallback)
```bash
cp .env.example .env          # add your OPENAI_API_KEY inside
python main.py --resumes ./resumes --jd jd.txt
# → tries OpenAI, falls back to Ollama, then HuggingFace automatically
```

### 2c – Ollama only
```bash
ollama pull gemma:2b && ollama pull llama3
python main.py --resumes ./resumes --jd jd.txt --provider ollama
```

### 2d – HuggingFace only (fully offline)
```bash
python main.py --resumes ./resumes --jd jd.txt --provider hf
# downloads google/flan-t5-base on first run
```

---

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--resumes PATH` | required | Directory of resume files |
| `--jd FILE` | required | Job description file |
| `--output FILE` | `results.json` | JSON output path |
| `--provider` | `auto` | `auto` (API→Ollama→HF), `ollama`, `openai`, `hf` |
| `--parser MODEL` | `gemma:2b` | LLM-1 model name |
| `--scorer MODEL` | `llama3` | LLM-2 model name |
| `--topk K` | `3` | K for Precision@K / Recall@K |
| `--verbose` | off | Debug logging |
| `--demo` | off | Mock LLM (no server needed) |
| `--api-key KEY` | env | OpenAI API key (alternative to `OPENAI_API_KEY` env var) |

---

## Scoring Weights

| Component | Weight | Description |
|-----------|--------|-------------|
| Skills | **50%** | Keyword + semantic skill match |
| Experience | **30%** | Years and relevance of work history |
| Education | **20%** | Qualification level vs requirement |

### Hybrid Matching Blend

| Signal | Weight |
|--------|--------|
| Keyword overlap (deterministic) | 30% |
| LLM semantic scoring | 70% |

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Precision@K** | Fraction of top-K candidates that are relevant |
| **Recall@K** | Fraction of relevant candidates captured in top-K |
| **MRR** | Mean Reciprocal Rank of first relevant result |
| **NDCG@K** | Normalized Discounted Cumulative Gain |
| **Explainability Score** | Proxy quality score for LLM explanations |
| **Approach Comparison** | Keyword-only vs LLM-only vs Hybrid |
| **Latency** | Parse time, match time, total time |

---

## Output

### Terminal
```
=================================================================
  CV SORTING PIPELINE – Results
=================================================================
  Job Title   : Senior Machine Learning Engineer
  Candidates  : 5
=================================================================

Ranked Candidates:

  #1  Alice Chen               Score:   91.8/100  Grade: A  [Strong Hire]
       Skills: 92  Experience: 95  Education: 100  Keyword: 85
       ✓ Matched: python, pytorch, langchain, mlflow, docker...
       Reasoning: Alice has all 12 required skills and exceeds
         experience requirement with 6 years of production LLM work.

  #2  Carol Nguyen             Score:   90.1/100  Grade: A  [Strong Hire]
  ...
```

### results.json
Full structured output including:
- Job description structured data
- All parsed resumes
- Ranked candidates with all component scores
- Complete evaluation metrics
- Model metadata and latency

---

## File Structure

All code files are in the **same flat directory** – no subdirectories for code.

```
Codebase/
├── main.py           Entry point + CLI + MockLLM + pipeline orchestrator
├── config.py         Central configuration (models, weights, thresholds)
├── llm_base.py       BaseLLM abstract interface (LLM-agnostic contract)
├── llm_ollama.py     OllamaLLM – local server (gemma:2b, llama3)
├── llm_openai.py     OpenAILLM – OpenAI API (gpt-3.5-turbo, gpt-4o)
├── llm_hf.py         HuggingFaceLLM – local Flan-T5 extraction layer
├── llm_fallback.py   FallbackLLM – cascading chain (OpenAI→Ollama→HF)
├── llm_factory.py    Factory: wires config/CLI flags → LLM instances
├── file_reader.py    PDF / DOCX / TXT text extraction
├── parser.py         LLM-1: ResumeParser + JDParser (structured extraction)
├── matcher.py        LLM-2: HybridMatcher (keyword overlap + LLM scoring)
├── ranker.py         Ranker – sorts by score, assigns grade + recommendation
├── reporter.py       Terminal display (colour-coded) + results.json output
├── metrics.py        Evaluator – Precision@K, Recall@K, MRR, NDCG, latency
├── requirements.txt  Python dependencies
├── execution.txt     Execution syntax
├── jd.txt            Sample job description
├── sample_output.txt Example terminal output
└── resumes/          Sample resume files (PDF / DOCX / TXT accepted)
    ├── alice_chen.txt
    ├── bob_martinez.txt
    ├── carol_nguyen.txt
    ├── david_okafor.txt
    └── eve_patel.txt
```

---

## Design Decisions

1. **Clean Architecture**: Each layer has a single responsibility and communicates via dataclasses (`ParsedResume`, `ParsedJD`, `MatchResult`, `RankedCandidate`), making unit testing trivial.

2. **LLM Interface Pattern**: `BaseLLM.generate(prompt) -> str` decouples the pipeline from any specific provider. Adding a new LLM (e.g. Anthropic Claude) requires only a new 30-line class.

3. **Hybrid Matching**: Pure keyword matching is fast but ignores synonyms; pure LLM scoring is slow and may hallucinate. The 30/70 blend gives the best of both: interpretable keyword signals anchoring semantically rich LLM judgements.

4. **Anti-Hallucination Prompts**: Prompts explicitly instruct the LLM to not assume missing data, penalise absent skills, and provide evidence-based explanations. Temperature is set to 0.0 for deterministic outputs.

5. **Graceful Degradation**: Every LLM call is wrapped in retry logic. If LLM-2 fails entirely, keyword scores + heuristics are used as a fallback so the pipeline always produces output.

6. **Config-Driven**: All weights, model names, and thresholds live in `config.py` and can be overridden via environment variables or CLI flags – no code changes needed.

---

## Methodology Summary (for Project Report)

### Problem
Manually shortlisting candidates from a large resume pool is slow, inconsistent, and subject to human bias. The goal is to automate ranked shortlisting while producing transparent, evidence-based explanations for every decision.

### Approach
A six-stage pipeline processes resumes end-to-end without human intervention:

**Stage 1 – Ingestion**: `file_reader.py` extracts raw text from PDF, DOCX, and TXT files using PyMuPDF and python-docx.

**Stage 2 – Structured Parsing (LLM-1)**: `parser.py` uses a LangChain `PromptTemplate` to instruct `gemma:2b` (a lightweight 2B-parameter local model) to extract structured JSON fields — skills, experience, education, contact info — from raw resume text. The same LLM parses the job description into required/preferred skills and minimum experience. A lightweight model is chosen here because structured extraction is a mechanical, well-defined task that does not require deep reasoning.

**Stage 3 – Hybrid Matching (LLM-2)**: `matcher.py` applies a two-step scoring approach:
- *Step A (Deterministic)*: Keyword overlap between candidate skills and JD skills (30% of final score). Compound skill strings (e.g. "PyTorch or TensorFlow") are split into individual tokens before matching. Fuzzy matching handles acronyms and substrings.
- *Step B (Semantic)*: A second LangChain `PromptTemplate` sends a detailed candidate-vs-JD brief to `llama3` (a 7B reasoning model), which returns numeric sub-scores for skills (50%), experience (30%), and education (20%), plus a 2–4 sentence evidence-based explanation. A reasoning model is required here because the task demands nuanced judgement (e.g. "Is 3 years of backend engineering equivalent to 3 years of ML engineering?").
- *Blend*: `final_score = 0.30 × keyword_score + 0.70 × llm_overall_score`

**Stage 4 – Ranking**: `ranker.py` sorts candidates by `final_score` descending and assigns letter grades (A–F) and hiring recommendations (Strong Hire / Consider / Weak Consider / Reject).

**Stage 5 – Evaluation**: `metrics.py` computes Precision@K, Recall@K, MRR, NDCG@K, and an explainability proxy score. Relevance labels are self-supervised: any candidate scoring ≥ 60/100 is considered "relevant". Latency and approach comparison (keyword-only vs LLM-only vs hybrid) are also reported.

**Stage 6 – Output**: `reporter.py` renders a colour-coded terminal leaderboard and saves the full structured output to `results.json`.

### Anti-Hallucination Measures
- LLM temperature set to 0.0 for deterministic, reproducible outputs.
- Prompts explicitly instruct the model NOT to assume skills not listed.
- Scoring prompts penalise missing required skills by a fixed amount per gap.
- If LLM-2 fails entirely, keyword scores + heuristics are used as fallback.
- Partial JSON repair (`_repair_truncated_json`) recovers truncated outputs from small models hitting their token limit.

### Models Used
| Role | Local (Ollama) | Cloud (OpenAI) | Local (HuggingFace) |
|------|---------------|----------------|---------------------|
| LLM-1 Parser | `gemma:2b` | `gpt-3.5-turbo` | `google/flan-t5-base` |
| LLM-2 Scorer | `llama3` | `gpt-4o` | — |
