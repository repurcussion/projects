"""
reporter.py – Terminal display and JSON persistence layer.

Responsibilities
----------------
1. Print a formatted header with job title and candidate count
2. Print the ranked candidate leaderboard with colour-coded scores/grades
3. Print the evaluation metrics table
4. Persist the full pipeline output to results.json

ANSI colour codes are used when writing to a real TTY; they are
automatically disabled when output is piped/redirected.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ranker import RankedCandidate   # flat import
from parser import ParsedJD          # flat import

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ANSI colour helpers (auto-disabled on non-TTY output)
# ---------------------------------------------------------------------------

_USE_COLOUR = os.isatty(1)   # True only when writing to a real terminal

RESET  = "\033[0m"  if _USE_COLOUR else ""
BOLD   = "\033[1m"  if _USE_COLOUR else ""
GREEN  = "\033[92m" if _USE_COLOUR else ""
YELLOW = "\033[93m" if _USE_COLOUR else ""
CYAN   = "\033[96m" if _USE_COLOUR else ""
RED    = "\033[91m" if _USE_COLOUR else ""
GREY   = "\033[90m" if _USE_COLOUR else ""

# Map letter grade to its display colour
GRADE_COLOUR = {
    "A": GREEN,
    "B": CYAN,
    "C": YELLOW,
    "D": YELLOW,
    "F": RED,
}


# ---------------------------------------------------------------------------
# Reporter class
# ---------------------------------------------------------------------------

class Reporter:
    """
    Handles all terminal output and JSON file persistence for the pipeline.

    Parameters
    ----------
    output_path : str
        Path for the results JSON file (default: "results.json").
    """

    def __init__(self, output_path: str = "results.json"):
        self.output_path = output_path

    # ------------------------------------------------------------------
    # Public methods – called in sequence by main.py
    # ------------------------------------------------------------------

    def print_header(self, jd: ParsedJD, candidate_count: int) -> None:
        """
        Print a summary header before the ranked results.

        Displays job title, required skills, min experience, and candidate count.
        """
        print()
        print(f"{BOLD}{'=' * 65}{RESET}")
        print(f"{BOLD}  CV SORTING PIPELINE – Results{RESET}")
        print(f"{BOLD}{'=' * 65}{RESET}")
        print(f"  Job Title   : {CYAN}{jd.job_title}{RESET}")
        print(f"  Required Skills : {', '.join(jd.required_skills[:6]) or 'N/A'}")
        print(f"  Min Experience  : {jd.min_experience_years} years")
        print(f"  Candidates      : {candidate_count}")
        print(f"  Timestamp       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{BOLD}{'=' * 65}{RESET}")
        print()

    def print_ranked_results(self, ranked: List[RankedCandidate]) -> None:
        """
        Print the ranked candidate leaderboard to stdout.

        For each candidate shows:
        - Rank, name, final score, grade, and recommendation
        - LLM sub-scores (skills, experience, education) and keyword score
        - Matched and missing skills (top 5 each)
        - Wrapped explanation from LLM-2
        """
        print(f"{BOLD}Ranked Candidates:{RESET}")
        print()

        for rc in ranked:
            # Choose colour based on grade
            grade_col = GRADE_COLOUR.get(rc.grade, RESET)
            # Choose colour based on recommendation strength
            rec_col = GREEN if "Hire" in rc.recommendation else (
                YELLOW if "Consider" in rc.recommendation else RED
            )

            # ---- Line 1: rank, name, score, grade, recommendation ----
            print(
                f"  {BOLD}#{rc.rank:<2}{RESET}  "
                f"{BOLD}{rc.candidate_name:<25}{RESET}  "
                f"Score: {grade_col}{rc.final_score:>6.1f}/100{RESET}  "
                f"Grade: {grade_col}{rc.grade}{RESET}  "
                f"[{rec_col}{rc.recommendation}{RESET}]"
            )

            # ---- Line 2: score breakdown ----
            print(
                f"       {GREY}Skills: {rc.llm_skills_score:.0f}  "
                f"Experience: {rc.llm_experience_score:.0f}  "
                f"Education: {rc.llm_education_score:.0f}  "
                f"Keyword: {rc.keyword_score:.0f}{RESET}"
            )

            # ---- Line 3: matched skills (max 5) ----
            if rc.matched_skills:
                print(
                    f"       {GREEN}✓ Matched:{RESET} "
                    f"{', '.join(rc.matched_skills[:5])}"
                    f"{'...' if len(rc.matched_skills) > 5 else ''}"
                )

            # ---- Line 4: missing skills (max 5) ----
            if rc.missing_skills:
                print(
                    f"       {RED}✗ Missing:{RESET} "
                    f"{', '.join(rc.missing_skills[:5])}"
                    f"{'...' if len(rc.missing_skills) > 5 else ''}"
                )

            # ---- Lines 5+: word-wrapped explanation ----
            if rc.explanation:
                words = rc.explanation.split()
                lines, line = [], []
                for w in words:
                    line.append(w)
                    if len(" ".join(line)) > 68:
                        lines.append(" ".join(line))
                        line = []
                if line:
                    lines.append(" ".join(line))
                print(f"       {GREY}Reasoning:{RESET}")
                for l in lines:
                    print(f"         {l}")

            print()   # blank line between candidates

        print(f"{BOLD}{'-' * 65}{RESET}")

    def print_evaluation_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Print the evaluation metrics table to stdout.

        Sections: core metrics, latency, approach comparison.
        """
        print()
        print(f"{BOLD}Evaluation Metrics:{RESET}")
        print(f"{BOLD}{'-' * 40}{RESET}")

        # Core IR metrics (Precision@K, Recall@K, MRR, NDCG@K, etc.)
        core = metrics.get("core_metrics", {})
        for key, val in core.items():
            label = key.replace("_", " ").title()
            if isinstance(val, float):
                print(f"  {label:<28}: {val:.4f}")
            else:
                print(f"  {label:<28}: {val}")

        # Pipeline latency breakdown
        latency = metrics.get("latency", {})
        if latency:
            print()
            print(
                f"  {'Latency (s)':<28}: "
                f"parse={latency.get('parse_time', 0):.2f}  "
                f"match={latency.get('match_time', 0):.2f}  "
                f"total={latency.get('total_time', 0):.2f}"
            )

        # Comparison of keyword-only vs LLM-only vs hybrid approaches
        comparison = metrics.get("approach_comparison", {})
        if comparison:
            print()
            print(f"  {'Approach Comparison':<28}")
            for approach, score in comparison.items():
                print(f"    {approach:<24}: {score:.2f}")

        print(f"{BOLD}{'-' * 40}{RESET}")

    def save_results(
        self,
        ranked: List[RankedCandidate],
        jd: ParsedJD,
        metrics: Dict[str, Any],
        parsed_resumes: Optional[List[Any]] = None,
    ) -> str:
        """
        Persist the full pipeline output to a JSON file.

        The output includes metadata, the job description, all ranked
        candidates with scores/explanations, evaluation metrics, and
        (optionally) the raw parsed resume data.

        Parameters
        ----------
        ranked : list of RankedCandidate
        jd : ParsedJD
        metrics : dict  (from Evaluator.compute_metrics)
        parsed_resumes : list of ParsedResume, optional

        Returns
        -------
        str  – absolute path of the saved file
        """
        output = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "job_title": jd.job_title,
                "total_candidates": len(ranked),
                "model_info": metrics.get("model_info", {}),
            },
            "job_description": jd.to_dict(),
            "ranked_candidates": [rc.to_dict() for rc in ranked],
            "evaluation_metrics": metrics,
        }

        # Include parsed resume details when available (useful for debugging)
        if parsed_resumes:
            output["parsed_resumes"] = [r.to_dict() for r in parsed_resumes]

        # Write to file with pretty-printing and unicode support
        path = Path(self.output_path)
        path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
        logger.info("Results saved to '%s'", path.resolve())
        return str(path.resolve())

    def save_charts(
        self,
        ranked: List[RankedCandidate],
        metrics: Dict[str, Any],
        output_dir: Optional[str] = "output",
    ) -> Optional[str]:
        """
        Generate and save graphical charts summarising pipeline results.

        Produces three PNG charts saved to *output_dir* (defaults to the
        directory of results.json):

        1. candidate_scores.png
           Horizontal bar chart of final score per candidate, colour-coded
           by grade (A=green, B=cyan, C/D=yellow, F=red).

        2. score_breakdown.png
           Grouped vertical bar chart showing Skills / Experience / Education
           sub-scores alongside the Keyword overlap score for each candidate.
           Visualises which dimension drives each candidate's ranking.

        3. approach_comparison.png
           Grouped bar chart comparing the mean scores of the three matching
           approaches: Keyword-only, LLM-only, and Hybrid.  Demonstrates
           that the hybrid approach combines the strengths of both signals.

        Requires matplotlib (pip install matplotlib).  If matplotlib is not
        installed the method logs a warning and returns None so the rest of
        the pipeline is unaffected.

        Parameters
        ----------
        ranked : list of RankedCandidate
            Sorted candidate list from Ranker.
        metrics : dict
            Evaluation metrics dict from Evaluator.compute_metrics().
        output_dir : str | None
            Directory to write PNG files into.  Defaults to the same
            directory as self.output_path (i.e. next to results.json).

        Returns
        -------
        str
            Directory path where charts were saved, or None on failure.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")   # non-interactive backend – no GUI needed
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            import numpy as np
        except ImportError:
            logger.warning(
                "matplotlib not installed – skipping chart generation. "
                "Install with: pip install matplotlib"
            )
            return None

        # Resolve output directory (default: "output/" subfolder next to results.json)
        save_dir = Path(output_dir) if output_dir else Path(self.output_path).parent / "output"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Shared colour map: grade → hex colour
        _GRADE_HEX = {"A": "#4caf50", "B": "#26c6da", "C": "#ffd54f",
                      "D": "#ffa726", "F": "#ef5350"}

        names    = [rc.candidate_name for rc in ranked]
        scores   = [rc.final_score    for rc in ranked]
        grades   = [rc.grade          for rc in ranked]
        bar_cols = [_GRADE_HEX.get(g, "#78909c") for g in grades]

        # ==============================================================
        # Chart 1 – Final Score Horizontal Bar Chart
        # ==============================================================
        fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.9)))

        # Plot bars right-to-left so rank 1 appears at top
        y_pos = range(len(names) - 1, -1, -1)
        bars = ax.barh(list(y_pos), scores, color=bar_cols, edgecolor="white",
                       height=0.6)

        # Score labels at end of each bar
        for bar, score in zip(bars, scores):
            ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
                    f"{score:.1f}", va="center", fontsize=9, color="#333333")

        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(names, fontsize=10)
        ax.set_xlim(0, 110)
        ax.set_xlabel("Final Score (out of 100)", fontsize=10)
        ax.set_title("Candidate Rankings – Final Score", fontsize=13, fontweight="bold",
                     pad=12)
        ax.axvline(x=60, color="#9e9e9e", linestyle="--", linewidth=1,
                   label="Relevance threshold (60)")
        ax.legend(fontsize=8, loc="lower right")
        ax.spines[["top", "right"]].set_visible(False)

        # Grade legend patches
        legend_patches = [
            mpatches.Patch(color=col, label=f"Grade {g}")
            for g, col in _GRADE_HEX.items()
        ]
        ax.legend(handles=legend_patches, fontsize=8,
                  loc="upper left" if scores[-1] > 40 else "upper right",
                  title="Grade", title_fontsize=8)

        fig.tight_layout()
        chart1_path = save_dir / "candidate_scores.png"
        fig.savefig(chart1_path, dpi=120)
        plt.close(fig)

        # ==============================================================
        # Chart 2 – Component Score Breakdown (grouped bars)
        # ==============================================================
        skills_scores  = [rc.llm_skills_score     for rc in ranked]
        exp_scores     = [rc.llm_experience_score  for rc in ranked]
        edu_scores     = [rc.llm_education_score   for rc in ranked]
        kw_scores      = [rc.keyword_score         for rc in ranked]

        x      = np.arange(len(names))
        width  = 0.2

        fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.6), 6))

        ax.bar(x - 1.5 * width, skills_scores, width, label="Skills (LLM)",
               color="#42a5f5", edgecolor="white")
        ax.bar(x - 0.5 * width, exp_scores,    width, label="Experience (LLM)",
               color="#66bb6a", edgecolor="white")
        ax.bar(x + 0.5 * width, edu_scores,    width, label="Education (LLM)",
               color="#ffa726", edgecolor="white")
        ax.bar(x + 1.5 * width, kw_scores,     width, label="Keyword Overlap",
               color="#ab47bc", edgecolor="white")

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
        ax.set_ylim(0, 115)
        ax.set_ylabel("Score (out of 100)", fontsize=10)
        ax.set_title("Score Breakdown by Component per Candidate",
                     fontsize=13, fontweight="bold", pad=12)
        ax.legend(fontsize=9)
        ax.axhline(y=60, color="#9e9e9e", linestyle="--", linewidth=1,
                   label="Relevance threshold")
        ax.spines[["top", "right"]].set_visible(False)

        fig.tight_layout()
        chart2_path = save_dir / "score_breakdown.png"
        fig.savefig(chart2_path, dpi=120)
        plt.close(fig)

        # ==============================================================
        # Chart 3 – Approach Comparison (Keyword vs LLM vs Hybrid)
        # ==============================================================
        comparison = metrics.get("approach_comparison", {})
        if comparison:
            approach_labels = []
            approach_means  = []
            approach_stds   = []
            approach_colors = ["#ab47bc", "#42a5f5", "#66bb6a"]

            # Extract mean and std for each approach in a defined order
            pairs = [
                ("keyword_only", "Keyword Only"),
                ("llm_only",     "LLM Only"),
                ("hybrid",       "Hybrid"),
            ]
            for key, label in pairs:
                mean_key = f"{key}_mean"
                std_key  = f"{key}_std"
                if mean_key in comparison:
                    approach_labels.append(label)
                    approach_means.append(comparison[mean_key])
                    approach_stds.append(comparison.get(std_key, 0))

            if approach_labels:
                fig, ax = plt.subplots(figsize=(6, 4.5))
                x_pos = np.arange(len(approach_labels))
                bars = ax.bar(x_pos, approach_means,
                              yerr=approach_stds,
                              color=approach_colors[:len(approach_labels)],
                              capsize=5, edgecolor="white", width=0.5,
                              error_kw={"ecolor": "#555", "elinewidth": 1.2})

                # Value labels on top of each bar
                for bar, mean in zip(bars, approach_means):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + max(approach_stds) + 1,
                            f"{mean:.1f}", ha="center", fontsize=10,
                            fontweight="bold", color="#333")

                ax.set_xticks(x_pos)
                ax.set_xticklabels(approach_labels, fontsize=11)
                ax.set_ylim(0, 105)
                ax.set_ylabel("Mean Score (out of 100)", fontsize=10)
                ax.set_title("Matching Approach Comparison\n"
                             "(error bars = ±1 std dev)",
                             fontsize=12, fontweight="bold", pad=10)
                ax.axhline(y=60, color="#9e9e9e", linestyle="--",
                           linewidth=1, label="Relevance threshold (60)")
                ax.legend(fontsize=8)
                ax.spines[["top", "right"]].set_visible(False)

                fig.tight_layout()
                chart3_path = save_dir / "approach_comparison.png"
                fig.savefig(chart3_path, dpi=120)
                plt.close(fig)

        logger.info("Charts saved to '%s'", save_dir.resolve())
        print(f"{BOLD}Charts saved to:{RESET} {CYAN}{save_dir.resolve()}{RESET}")
        return str(save_dir.resolve())

    def print_footer(self, saved_path: str) -> None:
        """Print closing summary showing where the results were saved."""
        print(f"{BOLD}Results saved to:{RESET} {CYAN}{saved_path}{RESET}")
        print()
