from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import json
import logging
from enum import Enum
import hashlib
import time
from collections import defaultdict
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== Core Data Models ====================

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

@dataclass
class HeadingInfo:
    """Structured heading information"""
    text: str
    level: int
    page: int
    bbox: Tuple[float, float, float, float]
    confidence: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'text': self.text,
            'level': self.level,
            'page': self.page,
            'bbox': self.bbox,
            'confidence': self.confidence,
            'metadata': self.metadata
        }

@dataclass
class DocumentOutline:
    """Complete document outline with metadata"""
    title: str
    headings: List[HeadingInfo]
    document_type: str
    processing_time: float
    confidence_score: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'title': self.title,
            'headings': [h.to_dict() for h in self.headings],
            'document_type': self.document_type,
            'processing_time': self.processing_time,
            'confidence_score': self.confidence_score,
            'metadata': self.metadata
        }

class ProcessingStatus(Enum):
    """Processing status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    CACHED = "cached"

# ==================== Strategy Pattern for Text Extraction ====================

class TextExtractionStrategy(ABC):
    """Abstract base class for text extraction strategies"""
    
    @abstractmethod
    def extract_spans(self, pdf_path: str) -> List[SpanInfo]:
        """Extract text spans from PDF"""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get strategy identifier"""
        pass

class DefaultTextExtractor(TextExtractionStrategy):
    """Default text extraction implementation"""
    
    def extract_spans(self, pdf_path: str) -> List[SpanInfo]:
        """Extract text spans using default method"""
        from pdf_loader import extract_text_spans
        return extract_text_spans(pdf_path)
    
    def get_strategy_name(self) -> str:
        return "default_extractor"

class OCRTextExtractor(TextExtractionStrategy):
    """OCR-based text extraction for scanned documents"""
    
    def extract_spans(self, pdf_path: str) -> List[SpanInfo]:
        """Extract text spans using OCR"""
        # Implementation would use OCR libraries
        logger.info(f"Using OCR extraction for {pdf_path}")
        # Placeholder implementation
        return []
    
    def get_strategy_name(self) -> str:
        return "ocr_extractor"

# ==================== Strategy Pattern for Heading Detection ====================

class HeadingDetectionStrategy(ABC):
    """Abstract base class for heading detection strategies"""
    
    @abstractmethod
    def detect_headings(self, spans: List[SpanInfo]) -> Tuple[str, List[HeadingInfo]]:
        """Detect headings and return title and heading list"""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get strategy identifier"""
        pass

class DefaultHeadingDetector(HeadingDetectionStrategy):
    """Default heading detection implementation"""
    
    def detect_headings(self, spans: List[SpanInfo]) -> Tuple[str, List[HeadingInfo]]:
        """Detect headings using default method"""
        from heading_detector import detect_headings_and_title
        
        title, outline = detect_headings_and_title(spans)
        
        # Convert to HeadingInfo objects
        headings = []
        for h in outline:
            heading = HeadingInfo(
                text=h['text'],
                level=self._parse_level(h.get('level', 1)),
                page=h['page'],
                bbox=h.get('bbox', (0, 0, 0, 0)),
                confidence=h.get('confidence', 0.5),
                metadata=h.get('metadata', {})
            )
            headings.append(heading)
        
        return title, headings
    
    def _parse_level(self, level: Any) -> int:
        """Parse heading level to integer"""
        if isinstance(level, int):
            return level
        if isinstance(level, str):
            if level.startswith('H'):
                return int(level[1:])
            return 1
        return 1
    
    def get_strategy_name(self) -> str:
        return "default_detector"

class EnhancedHeadingDetector(HeadingDetectionStrategy):
    """Enhanced heading detection using your ensemble system"""
    
    def __init__(self):
        # This would use your EnsembleHeadingDetector
        pass
    
    def detect_headings(self, spans: List[SpanInfo]) -> Tuple[str, List[HeadingInfo]]:
        """Detect headings using enhanced ensemble method"""
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Using enhanced heading detection (real implementation)")
        from heading_detector import detect_headings_and_title
        title, outline = detect_headings_and_title(spans)
        headings = []
        for h in outline:
            heading = HeadingInfo(
                text=h['text'],
                level=self._parse_level(h.get('level', 1)),
                page=h['page'],
                bbox=h.get('bbox', (0, 0, 0, 0)),
                confidence=h.get('confidence', 0.5),
                metadata=h.get('metadata', {})
            )
            headings.append(heading)
        return title, headings
    
    def _parse_level(self, level: Any) -> int:
        """Parse heading level to integer"""
        if isinstance(level, int):
            return level
        if isinstance(level, str) and level.startswith('H'):
            return int(level[1:])
        return 1
    
    def get_strategy_name(self) -> str:
        return "enhanced_detector"

# ==================== Deduplication Strategy ====================

class DeduplicationStrategy(ABC):
    """Abstract base class for deduplication strategies"""
    
    @abstractmethod
    def deduplicate(self, headings: List[HeadingInfo]) -> List[HeadingInfo]:
        """Remove duplicate headings"""
        pass

class BasicDeduplicator(DeduplicationStrategy):
    """Basic deduplication based on text, level, and page"""
    
    def deduplicate(self, headings: List[HeadingInfo]) -> List[HeadingInfo]:
        """Remove duplicates using basic strategy"""
        seen = set()
        deduplicated = []
        
        for heading in headings:
            # Create unique key
            key = (heading.text.strip().lower(), heading.level, heading.page)
            
            if key not in seen:
                deduplicated.append(heading)
                seen.add(key)
            else:
                logger.debug(f"Removed duplicate heading: {heading.text}")
        
        return deduplicated

class AdvancedDeduplicator(DeduplicationStrategy):
    """Advanced deduplication using similarity and context"""
    
    def __init__(self, similarity_threshold: float = 0.9):
        self.similarity_threshold = similarity_threshold
    
    def deduplicate(self, headings: List[HeadingInfo]) -> List[HeadingInfo]:
        """Remove duplicates using advanced similarity matching"""
        if not headings:
            return headings
        
        deduplicated = []
        
        for heading in headings:
            if not self._is_similar_to_existing(heading, deduplicated):
                deduplicated.append(heading)
            else:
                logger.debug(f"Removed similar heading: {heading.text}")
        
        return deduplicated
    
    def _is_similar_to_existing(self, heading: HeadingInfo, 
                              existing: List[HeadingInfo]) -> bool:
        """Check if heading is similar to any existing heading"""
        for existing_heading in existing:
            if self._calculate_similarity(heading, existing_heading) > self.similarity_threshold:
                return True
        return False
    
    def _calculate_similarity(self, h1: HeadingInfo, h2: HeadingInfo) -> float:
        """Calculate similarity between two headings"""
        # Simple similarity based on text overlap and position
        text_sim = self._text_similarity(h1.text, h2.text)
        page_sim = 1.0 if h1.page == h2.page else 0.0
        level_sim = 1.0 if h1.level == h2.level else 0.0
        
        return (text_sim * 0.6 + page_sim * 0.2 + level_sim * 0.2)
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0

# ==================== Caching System ====================

class CacheStrategy(ABC):
    """Abstract base class for caching strategies"""
    
    @abstractmethod
    def get(self, key: str) -> Optional[DocumentOutline]:
        """Get cached outline"""
        pass
    
    @abstractmethod
    def set(self, key: str, outline: DocumentOutline) -> None:
        """Cache outline"""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear cache"""
        pass

class InMemoryCache(CacheStrategy):
    """In-memory cache implementation"""
    
    def __init__(self, max_size: int = 100):
        self.cache: Dict[str, DocumentOutline] = {}
        self.max_size = max_size
        self.access_times: Dict[str, float] = {}
    
    def get(self, key: str) -> Optional[DocumentOutline]:
        """Get cached outline"""
        if key in self.cache:
            self.access_times[key] = time.time()
            logger.debug(f"Cache hit for key: {key}")
            return self.cache[key]
        
        logger.debug(f"Cache miss for key: {key}")
        return None
    
    def set(self, key: str, outline: DocumentOutline) -> None:
        """Cache outline with LRU eviction"""
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        self.cache[key] = outline
        self.access_times[key] = time.time()
        logger.debug(f"Cached outline for key: {key}")
    
    def clear(self) -> None:
        """Clear all cache"""
        self.cache.clear()
        self.access_times.clear()
        logger.info("Cache cleared")
    
    def _evict_lru(self) -> None:
        """Evict least recently used item"""
        if not self.access_times:
            return
        
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[lru_key]
        del self.access_times[lru_key]
        logger.debug(f"Evicted LRU item: {lru_key}")

# ==================== Observer Pattern for Progress Tracking ====================

class ProcessingObserver(ABC):
    """Abstract observer for processing events"""
    
    @abstractmethod
    def on_processing_start(self, pdf_path: str) -> None:
        """Called when processing starts"""
        pass
    
    @abstractmethod
    def on_processing_complete(self, pdf_path: str, outline: DocumentOutline) -> None:
        """Called when processing completes"""
        pass
    
    @abstractmethod
    def on_processing_error(self, pdf_path: str, error: Exception) -> None:
        """Called when processing fails"""
        pass

class LoggingObserver(ProcessingObserver):
    """Observer that logs processing events"""
    
    def on_processing_start(self, pdf_path: str) -> None:
        logger.info(f"Started processing: {pdf_path}")
    
    def on_processing_complete(self, pdf_path: str, outline: DocumentOutline) -> None:
        logger.info(f"Completed processing: {pdf_path} "
                   f"(found {len(outline.headings)} headings in {outline.processing_time:.2f}s)")
    
    def on_processing_error(self, pdf_path: str, error: Exception) -> None:
        logger.error(f"Error processing {pdf_path}: {error}")

class MetricsObserver(ProcessingObserver):
    """Observer that collects metrics"""
    
    def __init__(self):
        self.metrics = {
            'total_processed': 0,
            'successful_processed': 0,
            'failed_processed': 0,
            'total_processing_time': 0.0,
            'average_headings_per_doc': 0.0
        }
    
    def on_processing_start(self, pdf_path: str) -> None:
        self.metrics['total_processed'] += 1
    
    def on_processing_complete(self, pdf_path: str, outline: DocumentOutline) -> None:
        self.metrics['successful_processed'] += 1
        self.metrics['total_processing_time'] += outline.processing_time
        
        # Update average headings
        total_headings = sum(len(outline.headings) for outline in [outline])
        self.metrics['average_headings_per_doc'] = (
            total_headings / self.metrics['successful_processed']
        )
    
    def on_processing_error(self, pdf_path: str, error: Exception) -> None:
        self.metrics['failed_processed'] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return self.metrics.copy()

# ==================== Main Outline Builder with Dependency Injection ====================

class OutlineBuilderConfig:
    """Configuration for the outline builder"""
    
    def __init__(self):
        self.text_extraction_strategy: TextExtractionStrategy = DefaultTextExtractor()
        self.heading_detection_strategy: HeadingDetectionStrategy = DefaultHeadingDetector()
        self.deduplication_strategy: DeduplicationStrategy = BasicDeduplicator()
        self.cache_strategy: CacheStrategy = InMemoryCache()
        self.observers: List[ProcessingObserver] = [LoggingObserver()]
        self.enable_caching: bool = True
        self.enable_validation: bool = True
        self.timeout_seconds: int = 300

class PDFOutlineBuilder:
    """Main PDF outline builder with dependency injection and advanced features"""
    
    def __init__(self, config: Optional[OutlineBuilderConfig] = None):
        self.config = config or OutlineBuilderConfig()
        self._processing_status: Dict[str, ProcessingStatus] = {}
        self._processing_results: Dict[str, DocumentOutline] = {}
    
    def build_outline(self, pdf_path: str) -> DocumentOutline:
        """Build document outline with full pipeline"""
        start_time = time.time()
        
        try:
            # Notify observers
            for observer in self.config.observers:
                observer.on_processing_start(pdf_path)
            
            # Update status
            self._processing_status[pdf_path] = ProcessingStatus.PROCESSING
            
            # Check cache first
            if self.config.enable_caching:
                cached_outline = self._get_cached_outline(pdf_path)
                if cached_outline:
                    self._processing_status[pdf_path] = ProcessingStatus.CACHED
                    return cached_outline
            
            # Extract text spans
            spans = self.config.text_extraction_strategy.extract_spans(pdf_path)
            
            # Validate input
            if self.config.enable_validation:
                self._validate_spans(spans)
            
            # Detect headings
            title, headings = self.config.heading_detection_strategy.detect_headings(spans)
            
            # Deduplicate headings
            deduplicated_headings = self.config.deduplication_strategy.deduplicate(headings)
            
            # Create outline
            processing_time = time.time() - start_time
            outline = DocumentOutline(
                title=title,
                headings=deduplicated_headings,
                document_type=self._detect_document_type(spans),
                processing_time=processing_time,
                confidence_score=self._calculate_confidence_score(deduplicated_headings),
                metadata=self._build_metadata(pdf_path, spans, deduplicated_headings)
            )
            
            # Cache result
            if self.config.enable_caching:
                self._cache_outline(pdf_path, outline)
            
            # Store result
            self._processing_results[pdf_path] = outline
            self._processing_status[pdf_path] = ProcessingStatus.COMPLETED
            
            # Notify observers
            for observer in self.config.observers:
                observer.on_processing_complete(pdf_path, outline)
            
            return outline
            
        except Exception as e:
            self._processing_status[pdf_path] = ProcessingStatus.ERROR
            
            # Notify observers
            for observer in self.config.observers:
                observer.on_processing_error(pdf_path, e)
            
            raise OutlineBuilderException(f"Failed to build outline for {pdf_path}: {e}") from e
    
    def build_outline_async(self, pdf_path: str) -> str:
        """Start asynchronous outline building"""
        processing_id = self._generate_processing_id(pdf_path)
        self._processing_status[processing_id] = ProcessingStatus.PENDING
        
        # In a real implementation, this would use threading/asyncio
        # For demo purposes, we'll just return the ID
        return processing_id
    
    def get_processing_status(self, pdf_path: str) -> ProcessingStatus:
        """Get current processing status"""
        return self._processing_status.get(pdf_path, ProcessingStatus.PENDING)
    
    def get_processing_result(self, pdf_path: str) -> Optional[DocumentOutline]:
        """Get processing result if available"""
        return self._processing_results.get(pdf_path)
    
    def batch_process(self, pdf_paths: List[str]) -> Dict[str, DocumentOutline]:
        """Process multiple PDFs in batch"""
        results = {}
        
        for pdf_path in pdf_paths:
            try:
                outline = self.build_outline(pdf_path)
                results[pdf_path] = outline
            except Exception as e:
                logger.error(f"Failed to process {pdf_path}: {e}")
                continue
        
        return results
    
    def _get_cached_outline(self, pdf_path: str) -> Optional[DocumentOutline]:
        """Get cached outline"""
        cache_key = self._generate_cache_key(pdf_path)
        return self.config.cache_strategy.get(cache_key)
    
    def _cache_outline(self, pdf_path: str, outline: DocumentOutline) -> None:
        """Cache outline"""
        cache_key = self._generate_cache_key(pdf_path)
        self.config.cache_strategy.set(cache_key, outline)
    
    def _generate_cache_key(self, pdf_path: str) -> str:
        """Generate cache key for PDF"""
        # Include file modification time and size in key
        import os
        try:
            stat = os.stat(pdf_path)
            key_data = f"{pdf_path}_{stat.st_mtime}_{stat.st_size}"
            return hashlib.md5(key_data.encode()).hexdigest()
        except OSError:
            return hashlib.md5(pdf_path.encode()).hexdigest()
    
    def _generate_processing_id(self, pdf_path: str) -> str:
        """Generate unique processing ID"""
        return f"{pdf_path}_{int(time.time())}"
    
    def _validate_spans(self, spans: List[SpanInfo]) -> None:
        """Validate extracted spans"""
        if not spans:
            raise ValidationError("No text spans extracted from PDF")
        
        for span in spans:
            if not span.text.strip():
                continue  # Skip empty spans
            
            if span.page < 1:
                raise ValidationError(f"Invalid page number: {span.page}")
            
            if span.size <= 0:
                raise ValidationError(f"Invalid font size: {span.size}")
    
    def _detect_document_type(self, spans: List[SpanInfo]) -> str:
        """Detect document type based on content"""
        text_content = ' '.join([span.text for span in spans]).lower()
        
        # Simple heuristics
        if any(keyword in text_content for keyword in ['abstract', 'methodology', 'references']):
            return 'academic'
        elif any(keyword in text_content for keyword in ['api', 'configuration', 'installation']):
            return 'technical'
        elif any(keyword in text_content for keyword in ['executive summary', 'findings']):
            return 'report'
        else:
            return 'document'
    
    def _calculate_confidence_score(self, headings: List[HeadingInfo]) -> float:
        """Calculate overall confidence score"""
        if not headings:
            return 0.0
        
        return sum(h.confidence for h in headings) / len(headings)
    
    def _build_metadata(self, pdf_path: str, spans: List[SpanInfo], 
                       headings: List[HeadingInfo]) -> Dict[str, Any]:
        """Build metadata for the outline"""
        return {
            'pdf_path': pdf_path,
            'total_spans': len(spans),
            'total_headings': len(headings),
            'pages_processed': len(set(span.page for span in spans)),
            'extraction_strategy': self.config.text_extraction_strategy.get_strategy_name(),
            'detection_strategy': self.config.heading_detection_strategy.get_strategy_name(),
            'deduplication_strategy': type(self.config.deduplication_strategy).__name__,
            'timestamp': time.time()
        }

# ==================== Custom Exceptions ====================

class OutlineBuilderException(Exception):
    """Base exception for outline builder"""
    pass

class ValidationError(OutlineBuilderException):
    """Validation error"""
    pass

class ProcessingTimeoutError(OutlineBuilderException):
    """Processing timeout error"""
    pass

# ==================== Factory Pattern for Easy Configuration ====================

class OutlineBuilderFactory:
    """Factory for creating pre-configured outline builders"""
    
    @staticmethod
    def create_basic_builder() -> PDFOutlineBuilder:
        """Create basic outline builder"""
        config = OutlineBuilderConfig()
        return PDFOutlineBuilder(config)
    
    @staticmethod
    def create_enhanced_builder() -> PDFOutlineBuilder:
        """Create enhanced outline builder with advanced features"""
        config = OutlineBuilderConfig()
        config.heading_detection_strategy = EnhancedHeadingDetector()
        config.deduplication_strategy = AdvancedDeduplicator()
        config.observers.append(MetricsObserver())
        return PDFOutlineBuilder(config)
    
    @staticmethod
    def create_high_performance_builder() -> PDFOutlineBuilder:
        """Create high-performance builder with caching and metrics"""
        config = OutlineBuilderConfig()
        config.cache_strategy = InMemoryCache(max_size=500)
        config.deduplication_strategy = AdvancedDeduplicator()
        config.observers = [LoggingObserver(), MetricsObserver()]
        config.enable_caching = True
        return PDFOutlineBuilder(config)

# ==================== Usage Examples ====================

def demo_usage():
    """Demonstrate various usage patterns"""
    
    print("=== Enhanced PDF Outline Builder Demo ===\n")
    
    # 1. Basic usage
    print("1. Basic Usage:")
    builder = OutlineBuilderFactory.create_basic_builder()
    print("\n2. Enhanced Usage:")
    config = OutlineBuilderConfig()
    config.deduplication_strategy = AdvancedDeduplicator(similarity_threshold=0.8)
    config.observers.append(MetricsObserver())
    
    enhanced_builder = PDFOutlineBuilder(config)
    
    print("\n3. Batch Processing:")
    pdf_files = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]

    
    # 4. Metrics collection
    print("\n4. Metrics Collection:")
    metrics_observer = MetricsObserver()
    config.observers.append(metrics_observer)
    
    print("\nDemo completed!")

if __name__ == "__main__":
    demo_usage() 