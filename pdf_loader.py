import fitz
from typing import List, Optional
import logging
from dataclasses import dataclass
from typing import Tuple

# Configure logging
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class SpanInfo:
    """Immutable data class for text span information"""
    text: str
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    size: float
    page: int
    font: str = ""
    is_bold: bool = False
    is_italic: bool = False
    color: str = ""
    
    def __post_init__(self):
        # Validate data integrity
        if self.page < 1:
            raise ValueError("Page number must be positive")
        if self.size <= 0:
            raise ValueError("Font size must be positive")
        if len(self.bbox) != 4:
            raise ValueError("BBox must have 4 coordinates")

class PDFTextExtractor:
    """Enhanced PDF text extraction with robust error handling and filtering"""
    def __init__(self, min_font_size: float = 6.0, max_font_size: float = 72.0):
        self.min_font_size = min_font_size
        self.max_font_size = max_font_size

    def extract_text_spans(self, pdf_path: str) -> List[SpanInfo]:
        try:
            with fitz.open(pdf_path) as doc:
                spans = []
                logger.info(f"Extracting text spans from {pdf_path} ({doc.page_count} pages)")
                for page_num, page in enumerate(doc, 1):
                    try:
                        page_spans = self._extract_page_spans(page, page_num)
                        logger.debug(f"Page {page_num}: Extracted {len(page_spans)} spans")
                        spans.extend(page_spans)
                        if page_num % 10 == 0:
                            logger.debug(f"Processed {page_num}/{doc.page_count} pages")
                    except Exception as e:
                        logger.warning(f"Error processing page {page_num}: {e}")
                        continue
                logger.info(f"Total extracted spans before filtering: {len(spans)}")
                if spans:
                    logger.info(f"First 5 spans: {[s.text for s in spans[:5]]}")
                filtered_spans = self._filter_spans(spans)
                logger.info(f"Extracted {len(filtered_spans)} valid text spans from {doc.page_count} pages after filtering")
                if filtered_spans:
                    logger.info(f"First 5 filtered spans: {[s.text for s in filtered_spans[:5]]}")
                return filtered_spans
        except FileNotFoundError:
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to extract text from {pdf_path}: {e}")

    def _extract_page_spans(self, page: fitz.Page, page_num: int) -> List[SpanInfo]:
        spans = []
        try:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" not in block:
                    continue
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if not text or text.isspace():
                            continue
                        font_info = self._parse_font_info(span)
                        color = self._parse_color(span)
                        span_info = SpanInfo(
                            text=text,
                            bbox=tuple(span["bbox"]),
                            size=span["size"],
                            page=page_num,
                            font=font_info["font_name"],
                            is_bold=font_info["is_bold"],
                            is_italic=font_info["is_italic"],
                            color=color
                        )
                        spans.append(span_info)
        except Exception as e:
            logger.warning(f"Error extracting spans from page {page_num}: {e}")
        return spans

    def _parse_font_info(self, span: dict) -> dict:
        font_name = span.get("font", "")
        flags = span.get("flags", 0)
        is_bold = bool(flags & 16) or "bold" in font_name.lower()
        is_italic = bool(flags & 2) or "italic" in font_name.lower()
        return {
            "font_name": font_name,
            "is_bold": is_bold,
            "is_italic": is_italic,
            "flags": flags
        }

    def _parse_color(self, span: dict) -> str:
        color = span.get("color", 0)
        if color == 0:
            return "black"
        try:
            return f"#{color:06x}"
        except (ValueError, TypeError):
            return "black"

    def _filter_spans(self, spans: List[SpanInfo]) -> List[SpanInfo]:
        filtered = []
        for span in spans:
            if not (self.min_font_size <= span.size <= self.max_font_size):
                logger.debug(f"Skipping span with font size {span.size}: {span.text[:50]}...")
                continue
            if len(span.text) < 2:
                continue
            if self._is_mostly_special_chars(span.text):
                continue
            if not self._is_valid_bbox(span.bbox):
                logger.debug(f"Skipping span with invalid bbox {span.bbox}: {span.text[:50]}...")
                continue
            filtered.append(span)
        return filtered

    def _is_mostly_special_chars(self, text: str) -> bool:
        if len(text) < 3:
            return False
        alphanumeric_count = sum(1 for c in text if c.isalnum())
        return alphanumeric_count / len(text) < 0.3

    def _is_valid_bbox(self, bbox: Tuple[float, float, float, float]) -> bool:
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            return False
        if any(coord < 0 or coord > 10000 for coord in bbox):
            return False
        return True

def extract_text_spans(pdf_path: str) -> list:
    extractor = PDFTextExtractor()
    return extractor.extract_text_spans(pdf_path) 