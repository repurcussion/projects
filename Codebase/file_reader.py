"""
file_reader.py – Extracts raw text from PDF, DOCX, and TXT resume files.

Supported formats
-----------------
- .pdf   → PyMuPDF (fitz)
- .docx  → python-docx
- .txt   → plain UTF-8 text

All public functions return str.  Errors are logged and callers receive
an empty string on failure so the pipeline can continue without crashing.
"""

import logging
import os
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level format-specific extractors
# ---------------------------------------------------------------------------

def _extract_pdf(path: Path) -> str:
    """
    Extract plain text from a PDF file using PyMuPDF (fitz).

    Returns empty string if PyMuPDF is not installed or the file is corrupt.
    """
    try:
        import fitz  # PyMuPDF – install with: pip install pymupdf
        doc = fitz.open(str(path))
        pages = [page.get_text() for page in doc]
        doc.close()
        return "\n".join(pages).strip()
    except ImportError:
        logger.error("PyMuPDF not installed. Install with: pip install pymupdf")
        return ""
    except Exception as exc:
        logger.error("Failed to read PDF '%s': %s", path, exc)
        return ""


def _extract_docx(path: Path) -> str:
    """
    Extract plain text from a DOCX file using python-docx.

    Returns empty string if python-docx is not installed or the file is corrupt.
    """
    try:
        from docx import Document  # install with: pip install python-docx
        doc = Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs).strip()
    except ImportError:
        logger.error("python-docx not installed. Install with: pip install python-docx")
        return ""
    except Exception as exc:
        logger.error("Failed to read DOCX '%s': %s", path, exc)
        return ""


def _extract_txt(path: Path) -> str:
    """
    Read a plain-text file with UTF-8 encoding.

    Replaces undecodable bytes to avoid crashes on malformed files.
    """
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except Exception as exc:
        logger.error("Failed to read TXT '%s': %s", path, exc)
        return ""


# ---------------------------------------------------------------------------
# FileReader – unified dispatcher
# ---------------------------------------------------------------------------

class FileReader:
    """
    Unified file reader that dispatches to the correct format extractor.

    Usage
    -----
    reader = FileReader()
    text   = reader.read("/path/to/resume.pdf")
    """

    # File extensions supported for both resumes and job description files
    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt"}

    def read(self, path) -> str:
        """
        Extract and return plain text from *path*.

        Dispatches to the correct extractor based on file extension.
        Returns empty string if the file is unreadable or unsupported.
        """
        p = Path(path)

        if not p.exists():
            logger.error("File not found: '%s'", p)
            return ""

        if not p.is_file():
            logger.error("Path is not a file: '%s'", p)
            return ""

        ext = p.suffix.lower()

        # Warn early if the extension is not in the supported set.
        # We still attempt extraction so callers are never silently blocked.
        if ext not in self.SUPPORTED_EXTENSIONS:
            logger.warning(
                "Extension '%s' is not officially supported (supported: %s). "
                "Attempting plain-text read for '%s'.",
                ext,
                ", ".join(sorted(self.SUPPORTED_EXTENSIONS)),
                p.name,
            )

        # Dispatch to format-specific extractor
        if ext == ".pdf":
            text = _extract_pdf(p)
        elif ext in (".docx", ".doc"):
            text = _extract_docx(p)
        elif ext == ".txt":
            text = _extract_txt(p)
        else:
            # Unsupported extension – fall back to plain-text read
            text = _extract_txt(p)

        if not text:
            logger.warning("Extracted empty text from '%s'.", p)

        return text

    def read_directory(self, directory) -> Dict[str, str]:
        """
        Read all supported files in *directory* and return a name→text dict.

        Parameters
        ----------
        directory : str | Path
            Directory to scan for resume files.

        Returns
        -------
        dict
            Mapping {filename_stem: extracted_text} for all readable files.

        Raises
        ------
        FileNotFoundError
            If *directory* does not exist.
        ValueError
            If no readable resume files are found.
        """
        d = Path(directory)
        if not d.exists() or not d.is_dir():
            raise FileNotFoundError(
                f"Resume directory not found: '{d}'. "
                "Create the directory and place resume files inside."
            )

        results: Dict[str, str] = {}
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                text = self.read(f)
                if text:
                    results[f.stem] = text
                    logger.info("Read resume: '%s' (%d chars)", f.name, len(text))
                else:
                    logger.warning("Skipping empty file: '%s'", f.name)

        if not results:
            raise ValueError(
                f"No readable resume files found in '{d}'. "
                f"Supported formats: {', '.join(self.SUPPORTED_EXTENSIONS)}"
            )

        return results


# ---------------------------------------------------------------------------
# Module-level convenience functions (used by main.py)
# ---------------------------------------------------------------------------

_reader = FileReader()  # shared singleton instance


def read_resumes(directory) -> Dict[str, str]:
    """
    Read all resumes from *directory*.

    Returns
    -------
    dict
        {candidate_name_stem: raw_text}
    """
    return _reader.read_directory(directory)


def read_job_description(filepath) -> str:
    """
    Read the job description from *filepath*.

    Supported formats (same as resumes)
    ------------------------------------
    - .txt   Plain text
    - .pdf   PDF document (requires PyMuPDF: pip install pymupdf)
    - .docx  Word document (requires python-docx: pip install python-docx)
    - .doc   Legacy Word document (treated as .docx)

    Parameters
    ----------
    filepath : str | Path
        Path to the JD file in any supported format.

    Returns
    -------
    str
        Extracted plain text of the job description.

    Raises
    ------
    ValueError
        If the file is empty or unreadable.
    """
    text = _reader.read(filepath)
    if not text:
        raise ValueError(
            f"Job description file '{filepath}' is empty or unreadable."
        )
    return text
