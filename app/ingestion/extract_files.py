"""
PDF content extraction using Docling.

Supports OCR, table structure recognition, and image description
extraction from PDF documents.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Tuple
from dataclasses import dataclass

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PDFExtractionConfig:
    """Configuration for PDF extraction pipeline."""
    enable_ocr: bool = True
    images_scale: float = 2.0
    include_images: bool = True
    include_tables: bool = True


class PDFExtractor:
    """Extract structured content from PDF files using Docling."""

    def __init__(self, config: PDFExtractionConfig = None):
        self.config = config or PDFExtractionConfig()
        self._setup_converter()

    def _setup_converter(self):
        """Initialise the Docling document converter."""
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = self.config.enable_ocr
        pipeline_options.do_picture_description = self.config.include_images
        pipeline_options.do_table_structure = self.config.include_tables
        pipeline_options.images_scale = self.config.images_scale

        try:
            self.converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options
                    )
                }
            )
        except Exception as exc:
            logger.error("Failed to initialise Docling converter: %s", exc)
            raise

    def extract_pdf_content(self, pdf_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Extract text, tables, and image descriptions from a PDF.

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

        result = self.converter.convert(str(pdf_path))
        elapsed = time.time() - start_time

        doc = result.document
        content_text = doc.export_to_markdown()

        metadata = {
            "source": str(pdf_path),
            "title": pdf_path.stem,
            "processing_time": round(elapsed, 2),
            "pages": len(doc.pages),
            "texts": len(doc.texts),
            "pictures": len(doc.pictures),
            "tables": len(doc.tables),
            "extraction_method": "docling",
            "content_type": "pdf",
        }
        return content_text, metadata


def create_pdf_extractor(config: PDFExtractionConfig = None) -> PDFExtractor:
    """Factory function for ``PDFExtractor``."""
    return PDFExtractor(config)
