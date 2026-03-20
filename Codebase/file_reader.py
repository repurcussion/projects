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
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Secure remote file retrieval (SAS / presigned URLs)
# ---------------------------------------------------------------------------

def _download_url(url: str) -> Optional[Path]:
    """
    Fetch a remote resume file from *url* into a transient local temp file.

    Supports:
    - Azure Blob Storage SAS URLs
    - AWS S3 presigned URLs
    - Any plain HTTPS direct-download link

    The caller is responsible for unlinking the returned Path after use.
    Returns None on any network or HTTP error.
    """
    try:
        import requests
    except ImportError:
        logger.error("requests not installed. Install with: pip install requests")
        return None

    try:
        logger.info("Fetching remote resume: %s", url[:80])
        resp = requests.get(url, timeout=30, allow_redirects=True)
        resp.raise_for_status()

        # Infer file extension from Content-Type header or bare URL path
        content_type = resp.headers.get("Content-Type", "").lower()
        bare_url = url.split("?")[0].lower()   # strip SAS query string
        if "pdf" in content_type or bare_url.endswith(".pdf"):
            suffix = ".pdf"
        elif "docx" in content_type or bare_url.endswith((".docx", ".doc")):
            suffix = ".docx"
        else:
            suffix = ".txt"

        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.write(resp.content)
        tmp.flush()
        tmp.close()
        logger.info(
            "Downloaded %d bytes → temp file '%s'", len(resp.content), tmp.name
        )
        return Path(tmp.name)
    except Exception as exc:
        logger.error("Failed to download URL '%s …': %s", url[:60], exc)
        return None


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

        *path* may be a local filesystem path **or** an HTTP/HTTPS URL
        (including Azure SAS and AWS presigned URLs).  Remote files are
        downloaded transiently to a temp file and deleted immediately
        after extraction – nothing is persisted to disk.

        Dispatches to the correct extractor based on file extension.
        Returns empty string if the file is unreadable or unsupported.
        """
        path_str = str(path)

        # ── Remote URL (SAS / presigned / plain HTTPS) ──────────────────
        if path_str.startswith(("http://", "https://")):
            tmp_path = _download_url(path_str)
            if tmp_path is None:
                return ""
            try:
                return self.read(tmp_path)   # recurse with local temp path
            finally:
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

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

    def read_from_urls(self, urls: List[str]) -> Dict[str, str]:
        """
        Fetch and extract text from a list of remote resume URLs.

        Each URL may point to a PDF, DOCX, or TXT file hosted on any
        HTTPS server including Azure Blob Storage (SAS) or AWS S3
        (presigned URL).  Files are handled transiently – no local copy
        is retained after extraction.

        Parameters
        ----------
        urls : list of str
            Resume URLs.  The candidate name is derived from the URL
            basename (stripping query strings and extension).

        Returns
        -------
        dict
            {candidate_name: extracted_text} for all successfully read URLs.

        Raises
        ------
        ValueError
            If no URLs could be read successfully.
        """
        results: Dict[str, str] = {}
        for url in urls:
            text = self.read(url)
            if text:
                name = Path(url.split("?")[0]).stem or f"candidate_{len(results)+1}"
                results[name] = text
                logger.info("Read remote resume: '%s' (%d chars)", name, len(text))
            else:
                logger.warning("Skipping unreadable URL: '%s'", url[:80])

        if not results:
            raise ValueError(
                "No resume URLs could be read. "
                "Check that the URLs are valid and accessible."
            )
        return results

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


def read_resumes_from_urls(urls: List[str]) -> Dict[str, str]:
    """
    Fetch and extract resumes from a list of remote URLs.

    Supports Azure SAS URLs, AWS S3 presigned URLs, and any HTTPS link.
    Files are processed transiently \u2013 no local copy is retained.

    Parameters
    ----------
    urls : list of str
        One URL per resume file.

    Returns
    -------
    dict
        {candidate_name: raw_text}
    """
    return _reader.read_from_urls(urls)


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
