import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import logging
import time
from pathlib import Path
from typing import Dict, Any, Tuple
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PDFExtractionConfig:
    """Configuration for PDF extraction pipeline."""
    enable_ocr: bool = False
    images_scale: float = 1.0
    include_images: bool = False
    include_tables: bool = False


class PDFExtractor:
    """Extract clean structured content from PDF files quickly and reliably."""

    def __init__(self, config: PDFExtractionConfig = None):
        self.config = config or PDFExtractionConfig()

    def extract_pdf_content(self, pdf_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Extract text and structure from a PDF.

        Args:
            pdf_path: Filesystem path to the PDF file.

        Returns:
            A tuple of (markdown_content, metadata_dict).
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        logger.info("Extracting content from: %s", pdf_path.name)
        start_time = time.time()

        content_parts = []
        page_count = 0

        # Method 1: PyMuPDF (Fastest, highest quality text & layout extraction)
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(pdf_path))
            page_count = len(doc)
            for page_num in range(page_count):
                page = doc.load_page(page_num)
                text = page.get_text("text")
                if text.strip():
                    content_parts.append(f"## Page {page_num + 1}\n\n{text.strip()}")
            doc.close()
            method = "pymupdf"
        except Exception as exc:
            logger.warning("PyMuPDF extraction failed (%s), falling back to pypdf", exc)
            try:
                import pypdf
                reader = pypdf.PdfReader(str(pdf_path))
                page_count = len(reader.pages)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text and text.strip():
                        content_parts.append(f"## Page {i + 1}\n\n{text.strip()}")
                method = "pypdf"
            except Exception as e2:
                logger.error("All PDF extraction methods failed: %s", e2)
                raise

        content_text = "\n\n".join(content_parts)
        elapsed = time.time() - start_time

        metadata = {
            "source": str(pdf_path),
            "title": pdf_path.stem,
            "processing_time": round(elapsed, 2),
            "pages": page_count,
            "extraction_method": method,
            "content_type": "pdf",
        }
        return content_text, metadata


def create_pdf_extractor(config: PDFExtractionConfig = None) -> PDFExtractor:
    """Factory function for ``PDFExtractor``."""
    return PDFExtractor(config)


def extract_file_content(file_path: str) -> Tuple[str, Dict[str, Any]]:
    """
    Extract content from any file: PDF, source code, Markdown, TXT, JSON, etc.

    Args:
        file_path: Filesystem path to the file.

    Returns:
        A tuple of (content_text, metadata_dict).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()

    # If it's a PDF, use the dedicated PDF extractor
    if ext == ".pdf":
        extractor = create_pdf_extractor()
        return extractor.extract_pdf_content(str(path))

    # Otherwise read as text/code
    start_time = time.time()
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1", errors="replace") as f:
            content = f.read()

    elapsed = time.time() - start_time
    code_extensions = {
        ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".cpp",
        ".h", ".hpp", ".cs", ".go", ".rs", ".php", ".rb", ".sql",
        ".sh", ".bash", ".json", ".yaml", ".yml", ".html", ".css",
        ".xml", ".toml", ".ini", ".env", ".vue", ".swift", ".kt",
    }
    is_code = ext in code_extensions

    metadata = {
        "source": str(path),
        "title": path.name,
        "processing_time": round(elapsed, 2),
        "pages": 1,
        "extraction_method": "direct_text",
        "content_type": "code" if is_code else "text",
        "file_extension": ext,
        "char_count": len(content),
    }
    return content, metadata

