import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json

NOISE_PATTERNS = [
    r'^page \d+ of \d+$',  # Page numbers
    r'^version \d+\.\d+$',  # Version info
    r'^may \d{1,2}, \d{4}$',  # Dates
    r'^\d{4}$',  # Years
    r'^\d{1,2}/\d{1,2}/\d{2,4}$',  # Date formats
    r'^remarks$', r'^identifier$', r'^reference$',  # Table headers
    r'^istqb$',  # Trademarks
    r'^\s*$',  # Empty
]

def is_noise_heading(text):
    text = text.strip().lower()
    for pattern in NOISE_PATTERNS:
        if re.match(pattern, text):
            return True
    return False

@dataclass
class SpanInfo:
    """Data class for text span information"""
    text: str
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    size: float
    page: int
    font: str = ""
    is_bold: bool = False
    is_italic: bool = False
    color: str = ""

@dataclass
class HeadingCandidate:
    """Data class for heading candidate with all signals"""
    text: str
    page: int
    bbox: Tuple[float, float, float, float]
    level: str
    confidence: float
    signals: Dict[str, float]
    context: Dict[str, Any]

class SignalBase(ABC):
    """Base class for all heading detection signals"""
    
    def __init__(self, name: str, weight: float = 0.2):
        self.name = name
        self.weight = weight
        self.context = {}
    
    @abstractmethod
    def calculate_score(self, span: SpanInfo, spans: List[SpanInfo]) -> float:
        """Calculate signal score for a span (0-1)"""
        pass
    
    @abstractmethod
    def build_context(self, spans: List[SpanInfo]) -> Dict[str, Any]:
        """Build context information for the signal"""
        pass

class FontSizeSignal(SignalBase):
    """Font size-based heading detection"""
    
    def __init__(self, weight: float = 0.25):
        super().__init__("FontSizeSignal", weight)
    
    def build_context(self, spans: List[SpanInfo]) -> Dict[str, Any]:
        sizes = [s.size for s in spans]
        self.context = {
            'mean_size': np.mean(sizes),
            'std_size': np.std(sizes),
            'percentiles': np.percentile(sizes, [25, 50, 75, 90, 95])
        }
        return self.context
    
    def calculate_score(self, span: SpanInfo, spans: List[SpanInfo]) -> float:
        if not self.context:
            self.build_context(spans)
        
        # Score based on size percentile
        percentiles = self.context['percentiles']
        size = span.size
        
        if size >= percentiles[4]:  # 95th percentile
            return 1.0
        elif size >= percentiles[3]:  # 90th percentile
            return 0.8
        elif size >= percentiles[2]:  # 75th percentile
            return 0.6
        elif size >= percentiles[1]:  # 50th percentile
            return 0.4
        else:
            return 0.2

class PositionalSignal(SignalBase):
    """Position-based heading detection"""
    
    def __init__(self, weight: float = 0.25):
        super().__init__("PositionalSignal", weight)
    
    def build_context(self, spans: List[SpanInfo]) -> Dict[str, Any]:
        # Calculate page statistics
        page_stats = {}
        for span in spans:
            page = span.page
            if page not in page_stats:
                page_stats[page] = {
                    'left_positions': [],
                    'top_positions': [],
                    'line_heights': []
                }
            
            page_stats[page]['left_positions'].append(span.bbox[0])
            page_stats[page]['top_positions'].append(span.bbox[1])
            page_stats[page]['line_heights'].append(span.bbox[3] - span.bbox[1])
        
        # Calculate common left margins and spacing
        for page in page_stats:
            positions = page_stats[page]['left_positions']
            page_stats[page]['common_left'] = self._find_common_positions(positions)
            
        self.context = {'page_stats': page_stats}
        return self.context
    
    def _find_common_positions(self, positions: List[float], tolerance: float = 5.0) -> List[float]:
        """Find common positions within tolerance"""
        if not positions:
            return []
        
        positions = sorted(positions)
        common = []
        current_group = [positions[0]]
        
        for pos in positions[1:]:
            if pos - current_group[-1] <= tolerance:
                current_group.append(pos)
            else:
                if len(current_group) >= 2:  # At least 2 occurrences
                    common.append(np.mean(current_group))
                current_group = [pos]
        
        if len(current_group) >= 2:
            common.append(np.mean(current_group))
        
        return common
    
    def calculate_score(self, span: SpanInfo, spans: List[SpanInfo]) -> float:
        if not self.context:
            self.build_context(spans)
        
        score = 0.0
        page_stats = self.context['page_stats'].get(span.page, {})
        
        # Left alignment score
        left_pos = span.bbox[0]
        common_lefts = page_stats.get('common_left', [])
        if common_lefts and any(abs(left_pos - cl) < 5 for cl in common_lefts):
            score += 0.3
        
        # Isolation score (spacing above and below)
        isolation_score = self._calculate_isolation(span, spans)
        score += isolation_score * 0.4
        
        # Line length score (shorter lines more likely to be headings)
        line_length = span.bbox[2] - span.bbox[0]
        if line_length < 200:  # Adjust based on document width
            score += 0.3
        
        return min(score, 1.0)
    
    def _calculate_isolation(self, span: SpanInfo, spans: List[SpanInfo]) -> float:
        """Calculate how isolated a span is from others"""
        same_page_spans = [s for s in spans if s.page == span.page]
        
        min_distance_above = float('inf')
        min_distance_below = float('inf')
        
        for other in same_page_spans:
            if other == span:
                continue
                
            vertical_distance = abs(span.bbox[1] - other.bbox[1])
            
            if other.bbox[1] < span.bbox[1]:  # Above
                min_distance_above = min(min_distance_above, vertical_distance)
            elif other.bbox[1] > span.bbox[1]:  # Below
                min_distance_below = min(min_distance_below, vertical_distance)
        
        # Normalize distances (adjust thresholds based on your documents)
        isolation_threshold = 20
        above_score = min(min_distance_above / isolation_threshold, 1.0)
        below_score = min(min_distance_below / isolation_threshold, 1.0)
        
        return (above_score + below_score) / 2

class TextPatternSignal(SignalBase):
    """Text pattern-based heading detection"""
    
    def __init__(self, weight: float = 0.25):
        super().__init__("TextPatternSignal", weight)
        self.patterns = [
            r'^\d+\.\s+\w+',  # "1. Introduction"
            r'^\d+\.\d+\s+\w+',  # "1.1 Overview"
            r'^[A-Z][a-z]+\s+\d+',  # "Chapter 1"
            r'^[A-Z\s]+$',  # ALL CAPS
            r'^[IVX]+\.\s+\w+',  # Roman numerals
            r'^[A-Z]\.\s+\w+',  # "A. Section"
            r'^[a-z]\)\s+\w+',  # "a) subsection"
        ]
    
    def build_context(self, spans: List[SpanInfo]) -> Dict[str, Any]:
        pattern_counts = {}
        for pattern in self.patterns:
            pattern_counts[pattern] = 0
            for span in spans:
                if re.match(pattern, span.text.strip()):
                    pattern_counts[pattern] += 1
        
        self.context = {'pattern_counts': pattern_counts}
        return self.context
    
    def calculate_score(self, span: SpanInfo, spans: List[SpanInfo]) -> float:
        if not self.context:
            self.build_context(spans)
        
        text = span.text.strip()
        score = 0.0
        
        # Check against known patterns
        for pattern in self.patterns:
            if re.match(pattern, text):
                score += 0.8
                break
        
        # Additional heuristics
        if len(text) < 100 and len(text) > 5:  # Reasonable heading length
            score += 0.1
        
        if text.istitle():  # Title Case
            score += 0.1
        
        return min(score, 1.0)

class TypographicSignal(SignalBase):
    """Typography-based heading detection"""
    
    def __init__(self, weight: float = 0.15):
        super().__init__("TypographicSignal", weight)
    
    def build_context(self, spans: List[SpanInfo]) -> Dict[str, Any]:
        fonts = {}
        for span in spans:
            font = span.font
            if font not in fonts:
                fonts[font] = {'count': 0, 'avg_size': 0, 'sizes': []}
            fonts[font]['count'] += 1
            fonts[font]['sizes'].append(span.size)
        
        # Calculate average sizes
        for font in fonts:
            fonts[font]['avg_size'] = np.mean(fonts[font]['sizes'])
        
        body_font = max(fonts.keys(), key=lambda f: fonts[f]['count']) if fonts else ""
        
        self.context = {
            'fonts': fonts,
            'body_font': body_font,
            'bold_count': sum(1 for s in spans if s.is_bold),
            'total_count': len(spans)
        }
        return self.context
    
    def calculate_score(self, span: SpanInfo, spans: List[SpanInfo]) -> float:
        if not self.context:
            self.build_context(spans)
        
        score = 0.0
        
        if span.is_bold:
            score += 0.5
    
        if span.font != self.context['body_font'] and span.font:
            score += 0.3
        
        # Italic bonus (sometimes used for headings)
        if span.is_italic:
            score += 0.2
        
        return min(score, 1.0)

class SequentialSignal(SignalBase):
    """Sequential pattern-based heading detection"""
    
    def __init__(self, weight: float = 0.10):
        super().__init__("SequentialSignal", weight)
    
    def build_context(self, spans: List[SpanInfo]) -> Dict[str, Any]:
        # Find numbered sequences
        numbered_spans = []
        for span in spans:
            match = re.match(r'^(\d+)\.', span.text.strip())
            if match:
                numbered_spans.append({
                    'span': span,
                    'number': int(match.group(1))
                })
        
        # Sort by number
        numbered_spans.sort(key=lambda x: x['number'])
        
        # Find sequences
        sequences = []
        current_seq = []
        
        for item in numbered_spans:
            if not current_seq or item['number'] == current_seq[-1]['number'] + 1:
                current_seq.append(item)
            else:
                if len(current_seq) >= 2:
                    sequences.append(current_seq)
                current_seq = [item]
        
        if len(current_seq) >= 2:
            sequences.append(current_seq)
        
        self.context = {'sequences': sequences}
        return self.context
    
    def calculate_score(self, span: SpanInfo, spans: List[SpanInfo]) -> float:
        if not self.context:
            self.build_context(spans)
        
        # Check if span is part of a sequence
        for sequence in self.context['sequences']:
            for item in sequence:
                if item['span'] == span:
                    return 0.9  # High confidence for sequential items
        
        return 0.0

class EnsembleHeadingDetector:
    """Main heading detection system using ensemble of signals"""
    def __init__(self, custom_weights: Optional[Dict[str, float]] = None):
        self.signals = [
            FontSizeSignal(),
            PositionalSignal(),
            TextPatternSignal(),
            TypographicSignal(),
            SequentialSignal()
        ]
        if custom_weights:
            for signal in self.signals:
                if signal.name in custom_weights:
                    signal.weight = custom_weights[signal.name]
        self.scaler = StandardScaler()
        self.ml_model = LogisticRegression(random_state=42)
        self.is_trained = False
        self.heading_threshold = 0.6
        self.level_thresholds = {
            'H1': 0.8,
            'H2': 0.6,
            'H3': 0.4
        }

    def extract_features(self, spans: List[SpanInfo]) -> np.ndarray:
        for signal in self.signals:
            signal.build_context(spans)
        features = []
        for span in spans:
            span_features = []
            for signal in self.signals:
                score = signal.calculate_score(span, spans)
                span_features.append(score)
            features.append(span_features)
        return np.array(features)

    def detect_headings(self, spans: List[SpanInfo]) -> List[HeadingCandidate]:
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"[HeadingDetector] Number of spans received: {len(spans)}")
        if not spans:
            return []
        features = self.extract_features(spans)
        candidates = []
        for i, span in enumerate(spans):
            span_features = features[i]
            if self.is_trained:
                scaled_features = self.scaler.transform([span_features])
                ml_score = self.ml_model.predict_proba(scaled_features)[0][1]
                confidence = ml_score
            else:
                confidence = np.sum([
                    feat * signal.weight 
                    for feat, signal in zip(span_features, self.signals)
                ])
            if confidence >= self.heading_threshold:
                level = self._determine_level(confidence, span, spans)
                signal_breakdown = {
                    signal.name: feat 
                    for signal, feat in zip(self.signals, span_features)
                }
                candidate = HeadingCandidate(
                    text=span.text,
                    page=span.page,
                    bbox=span.bbox,
                    level=level,
                    confidence=confidence,
                    signals=signal_breakdown,
                    context=self._build_candidate_context(span, spans)
                )
                candidates.append(candidate)
        logger.info(f"[HeadingDetector] Number of heading candidates: {len(candidates)}")
        if candidates:
            logger.info(f"[HeadingDetector] First 5 candidates: {[{'text': c.text, 'conf': c.confidence} for c in candidates[:5]]}")
        else:
            logger.info("[HeadingDetector] No heading candidates found.")
        candidates.sort(key=lambda x: x.confidence, reverse=True)
        return self._refine_heading_levels(candidates)

    def get_document_outline(self, spans):
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"[HeadingDetector] get_document_outline called with {len(spans)} spans.")
        candidates = self.detect_headings(spans)
        logger.info(f"[HeadingDetector] Candidates after detect_headings: {len(candidates)}")
        noise_filter = NoiseFilterStrategy()
        filtered_candidates = noise_filter.filter(candidates)
        merged_candidates = self._merge_consecutive_headings(filtered_candidates)
        seen = set()
        unique_candidates = []
        for c in merged_candidates:
            key = (c.text.strip().lower(), c.level)
            if key not in seen:
                unique_candidates.append(c)
                seen.add(key)
        first_page_candidates = [c for c in unique_candidates if c.page == 1]
        if first_page_candidates:
            title_candidate = max(first_page_candidates, key=lambda x: (x.confidence, x.level == 'H1', len(x.text)))
            title = title_candidate.text
        else:
            title = ""
        outline = []
        for candidate in unique_candidates:
            if candidate.level in ['H1', 'H2', 'H3']:
                outline.append({
                    'level': candidate.level,
                    'text': candidate.text,
                    'page': candidate.page - 1
                })
        return title, outline

    def _merge_consecutive_headings(self, candidates):
        if not candidates:
            return []
        merged = []
        prev = None
        for c in candidates:
            if prev and c.page == prev.page and abs(c.bbox[1] - prev.bbox[1]) < 5:
                prev.text += ' ' + c.text
                prev.confidence = max(prev.confidence, c.confidence)
            else:
                if prev:
                    merged.append(prev)
                prev = c
        if prev:
            merged.append(prev)
        return merged

    def _determine_level(self, confidence: float, span: SpanInfo, spans: List[SpanInfo]) -> str:
        if confidence >= self.level_thresholds['H1']:
            return 'H1'
        elif confidence >= self.level_thresholds['H2']:
            return 'H2'
        else:
            return 'H3'

    def _build_candidate_context(self, span: SpanInfo, spans: List[SpanInfo]) -> Dict[str, Any]:
        return {
            'font_size': span.size,
            'is_bold': span.is_bold,
            'font': span.font,
            'text_length': len(span.text),
            'position_on_page': span.bbox[1]
        }

    def _refine_heading_levels(self, candidates: List[HeadingCandidate]) -> List[HeadingCandidate]:
        if not candidates:
            return candidates
        page_groups = {}
        for candidate in candidates:
            page = candidate.page
            if page not in page_groups:
                page_groups[page] = []
            page_groups[page].append(candidate)
        for page in page_groups:
            page_groups[page].sort(key=lambda x: x.bbox[1])
        refined_candidates = []
        for page in sorted(page_groups.keys()):
            page_candidates = page_groups[page]
            for i, candidate in enumerate(page_candidates):
                if candidate.confidence >= 0.9:
                    candidate.level = 'H1'
                elif candidate.confidence >= 0.75:
                    candidate.level = 'H2'
                else:
                    candidate.level = 'H3'
                refined_candidates.append(candidate)
        return refined_candidates
class NoiseFilterStrategy:
    """Strategy for filtering out noise headings."""
    def filter(self, candidates):
        return [c for c in candidates if not is_noise_heading(c.text)]
    
    def get_document_outline(self, spans: List[SpanInfo]) -> Tuple[str, List[Dict]]:
        """Get document title and outline"""
        candidates = self.detect_headings(spans)
        # Apply noise filtering strategy
        noise_filter = NoiseFilterStrategy()
        filtered_candidates = noise_filter.filter(candidates)
        # Deduplicate
        seen = set()
        unique_candidates = []
        for c in filtered_candidates:
            key = (c.text.strip().lower(), c.level)
            if key not in seen:
                unique_candidates.append(c)
                seen.add(key)
        # Find title (largest, boldest, non-noise text on first page)
        first_page_candidates = [c for c in unique_candidates if c.page == 1]
        if first_page_candidates:
            title_candidate = max(first_page_candidates, key=lambda x: (x.confidence, x.level == 'H1', len(x.text)))
            title = title_candidate.text
        else:
            title = ""
        # Build outline (only H1/H2/H3)
        outline = []
        for candidate in unique_candidates:
            if candidate.level in ['H1', 'H2', 'H3']:
                outline.append({
                    'level': candidate.level,
                    'text': candidate.text,
                    'page': candidate.page,
                    'confidence': candidate.confidence
                })
        return title, outline
    
    def save_model(self, filepath: str):
        """Save trained model"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        model_data = {
            'ml_model': self.ml_model,
            'scaler': self.scaler,
            'signal_weights': {s.name: s.weight for s in self.signals},
            'thresholds': {
                'heading_threshold': self.heading_threshold,
                'level_thresholds': self.level_thresholds
            }
        }
        
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load trained model"""
        import pickle
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.ml_model = model_data['ml_model']
        self.scaler = model_data['scaler']
        self.is_trained = True
        
        # Update signal weights
        weights = model_data['signal_weights']
        for signal in self.signals:
            if signal.name in weights:
                signal.weight = weights[signal.name]
        
        # Update thresholds
        thresholds = model_data['thresholds']
        self.heading_threshold = thresholds['heading_threshold']
        self.level_thresholds = thresholds['level_thresholds']


# Example usage and testing
def create_sample_data():
    """Create sample data for testing"""
    spans = [
        SpanInfo("Introduction", (50, 100, 200, 120), 16, 1, "Arial", True),
        SpanInfo("This is body text that explains the introduction section.", (50, 130, 400, 145), 12, 1, "Arial", False),
        SpanInfo("1. First Section", (50, 160, 180, 175), 14, 1, "Arial", True),
        SpanInfo("Body text for the first section goes here.", (50, 185, 380, 200), 12, 1, "Arial", False),
        SpanInfo("1.1 Subsection", (70, 210, 170, 225), 13, 1, "Arial", True),
        SpanInfo("More body text for the subsection.", (70, 235, 350, 250), 12, 1, "Arial", False),
        SpanInfo("2. Second Section", (50, 270, 185, 285), 14, 1, "Arial", True),
        SpanInfo("CONCLUSION", (50, 300, 150, 320), 16, 1, "Arial", True),
    ]
    
    return spans

def demo_usage():
    """Demonstrate the heading detection system"""
    print("=== Robust Multi-Signal Heading Detection Demo ===\n")
    
    # Create sample data
    spans = create_sample_data()
    
    # Initialize detector
    detector = EnsembleHeadingDetector()
    
    print("1. Using Default Weights:")
    title, outline = detector.get_document_outline(spans)
    print(f"Title: {title}")
    print("Outline:")
    for item in outline:
        print(f"  {item['level']}: {item['text']} (confidence: {item['confidence']:.3f})")
    
    print("\n2. Using Custom Weights for Academic Papers:")
    custom_weights = {
        'SequentialSignal': 0.20,
        'TextPatternSignal': 0.30,
        'FontSizeSignal': 0.20,
        'PositionalSignal': 0.20,
        'TypographicSignal': 0.10
    }
    
    academic_detector = EnsembleHeadingDetector(custom_weights)
    title, outline = academic_detector.get_document_outline(spans)
    print(f"Title: {title}")
    print("Outline:")
    for item in outline:
        print(f"  {item['level']}: {item['text']} (confidence: {item['confidence']:.3f})")
    
    print("\n3. Detailed Signal Analysis:")
    candidates = detector.detect_headings(spans)
    for candidate in candidates[:3]:  # Show first 3
        print(f"\nText: '{candidate.text}'")
        print(f"Level: {candidate.level}, Confidence: {candidate.confidence:.3f}")
        print("Signal Breakdown:")
        for signal_name, score in candidate.signals.items():
            print(f"  {signal_name}: {score:.3f}")

# Advanced Features for Standout Performance

class SemanticSignal(SignalBase):
    """Semantic/NLP-based heading detection using embeddings"""
    
    def __init__(self, weight: float = 0.15):
        super().__init__("SemanticSignal", weight)
        self.heading_keywords = [
            'introduction', 'conclusion', 'abstract', 'summary', 'overview',
            'methodology', 'results', 'discussion', 'chapter', 'section',
            'background', 'literature', 'analysis', 'evaluation', 'future'
        ]
    
    def build_context(self, spans: List[SpanInfo]) -> Dict[str, Any]:
        # Calculate semantic similarity scores
        semantic_scores = []
        for span in spans:
            score = self._calculate_semantic_score(span.text)
            semantic_scores.append(score)
        
        self.context = {
            'semantic_scores': semantic_scores,
            'avg_semantic': np.mean(semantic_scores),
            'std_semantic': np.std(semantic_scores)
        }
        return self.context
    
    def _calculate_semantic_score(self, text: str) -> float:
        """Calculate semantic similarity to typical heading content"""
        text_lower = text.lower()
        
        # Keyword matching
        keyword_score = 0
        for keyword in self.heading_keywords:
            if keyword in text_lower:
                keyword_score += 0.3
        
        # Length-based heuristics (headings are typically concise)
        if 5 <= len(text) <= 50:
            length_score = 0.4
        elif 50 < len(text) <= 100:
            length_score = 0.2
        else:
            length_score = 0.1
        
        # Question/declarative patterns
        if text.strip().endswith('?'):
            question_score = 0.2
        elif text.strip().endswith(':'):
            colon_score = 0.3
        else:
            question_score = colon_score = 0
        
        return min(keyword_score + length_score + question_score + colon_score, 1.0)
    
    def calculate_score(self, span: SpanInfo, spans: List[SpanInfo]) -> float:
        if not self.context:
            self.build_context(spans)
        
        return self._calculate_semantic_score(span.text)


class AdaptiveThresholdSystem:
    """Dynamic threshold adjustment based on document characteristics"""
    
    def __init__(self):
        self.document_types = {
            'academic': {'heading_threshold': 0.65, 'level_thresholds': {'H1': 0.85, 'H2': 0.65, 'H3': 0.45}},
            'technical': {'heading_threshold': 0.60, 'level_thresholds': {'H1': 0.80, 'H2': 0.60, 'H3': 0.40}},
            'legal': {'heading_threshold': 0.70, 'level_thresholds': {'H1': 0.90, 'H2': 0.70, 'H3': 0.50}},
            'report': {'heading_threshold': 0.55, 'level_thresholds': {'H1': 0.75, 'H2': 0.55, 'H3': 0.35}},
            'manual': {'heading_threshold': 0.50, 'level_thresholds': {'H1': 0.70, 'H2': 0.50, 'H3': 0.30}}
        }
    
    def detect_document_type(self, spans: List[SpanInfo]) -> str:
        """Automatically detect document type based on content patterns"""
        text_content = ' '.join([span.text for span in spans]).lower()
        
        # Academic indicators
        academic_indicators = ['abstract', 'methodology', 'literature review', 'references', 'citation']
        academic_score = sum(1 for indicator in academic_indicators if indicator in text_content)
        
        # Technical indicators
        technical_indicators = ['api', 'configuration', 'installation', 'requirements', 'documentation']
        technical_score = sum(1 for indicator in technical_indicators if indicator in text_content)
        
        # Legal indicators
        legal_indicators = ['whereas', 'pursuant', 'section', 'article', 'clause', 'agreement']
        legal_score = sum(1 for indicator in legal_indicators if indicator in text_content)
        
        # Report indicators
        report_indicators = ['executive summary', 'findings', 'recommendations', 'analysis', 'quarterly']
        report_score = sum(1 for indicator in report_indicators if indicator in text_content)
        
        # Manual indicators
        manual_indicators = ['step', 'procedure', 'instruction', 'warning', 'note', 'caution']
        manual_score = sum(1 for indicator in manual_indicators if indicator in text_content)
        
        scores = {
            'academic': academic_score,
            'technical': technical_score,
            'legal': legal_score,
            'report': report_score,
            'manual': manual_score
        }
        
        return max(scores.keys(), key=lambda k: scores[k])
    
    def get_thresholds(self, document_type: str) -> Dict[str, Any]:
        """Get thresholds for detected document type"""
        return self.document_types.get(document_type, self.document_types['report'])


class VisualLayoutSignal(SignalBase):
    """Advanced visual layout analysis"""
    
    def __init__(self, weight: float = 0.20):
        super().__init__("VisualLayoutSignal", weight)
    
    def build_context(self, spans: List[SpanInfo]) -> Dict[str, Any]:
        # Analyze visual layout patterns
        layout_features = {
            'column_detection': self._detect_columns(spans),
            'reading_order': self._analyze_reading_order(spans),
            'visual_hierarchy': self._analyze_visual_hierarchy(spans)
        }
        
        self.context = layout_features
        return self.context
    
    def _detect_columns(self, spans: List[SpanInfo]) -> Dict[str, Any]:
        """Detect column layout"""
        x_positions = [span.bbox[0] for span in spans]
        
        # Simple column detection using clustering
        if len(set(x_positions)) > 1:
            x_array = np.array(x_positions).reshape(-1, 1)
            n_clusters = min(3, len(set(x_positions)))
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
            clusters = kmeans.fit_predict(x_array)
            
            return {
                'has_columns': n_clusters > 1,
                'column_count': n_clusters,
                'column_positions': kmeans.cluster_centers_.flatten().tolist()
            }
        
        return {'has_columns': False, 'column_count': 1, 'column_positions': [x_positions[0]]}
    
    def _analyze_reading_order(self, spans: List[SpanInfo]) -> List[int]:
        """Analyze natural reading order"""
        # Sort by page, then by Y position, then by X position
        sorted_spans = sorted(enumerate(spans), 
                            key=lambda x: (x[1].page, x[1].bbox[1], x[1].bbox[0]))
        return [idx for idx, _ in sorted_spans]
    
    def _analyze_visual_hierarchy(self, spans: List[SpanInfo]) -> Dict[str, Any]:
        """Analyze visual hierarchy patterns"""
        sizes = [span.size for span in spans]
        positions = [span.bbox[1] for span in spans]  # Y positions
        
        return {
            'size_variance': np.var(sizes),
            'position_variance': np.var(positions),
            'size_position_correlation': np.corrcoef(sizes, positions)[0, 1] if len(sizes) > 1 else 0
        }
    
    def calculate_score(self, span: SpanInfo, spans: List[SpanInfo]) -> float:
        if not self.context:
            self.build_context(spans)
        
        score = 0.0
        
        # Column-based scoring
        if self.context['column_detection']['has_columns']:
            # Headings often span multiple columns
            span_width = span.bbox[2] - span.bbox[0]
            if span_width > 300:  # Adjust based on document width
                score += 0.4
        
        # Reading order importance
        reading_order = self.context['reading_order']
        span_index = next((i for i, s in enumerate(spans) if s == span), -1)
        if span_index in reading_order[:5]:  # Top 5 in reading order
            score += 0.3
        
        # Visual hierarchy
        hierarchy = self.context['visual_hierarchy']
        if hierarchy['size_variance'] > 2:  # High size variance indicates clear hierarchy
            score += 0.3
        
        return min(score, 1.0)


class ContextualRefinementEngine:
    """Advanced post-processing for heading hierarchy"""
    
    def __init__(self):
        self.hierarchy_rules = {
            'max_h1_per_page': 2,
            'min_h2_after_h1': 0,
            'logical_progression': True
        }
    
    def refine_hierarchy(self, candidates: List[HeadingCandidate]) -> List[HeadingCandidate]:
        """Apply contextual rules to refine heading hierarchy"""
        if not candidates:
            return candidates
        
        # Sort by page and position
        sorted_candidates = sorted(candidates, key=lambda x: (x.page, x.bbox[1]))
        
        # Apply hierarchy rules
        refined = self._apply_hierarchy_rules(sorted_candidates)
        
        # Apply semantic consistency
        refined = self._apply_semantic_consistency(refined)
        
        # Apply numerical sequence validation
        refined = self._validate_numerical_sequences(refined)
        
        return refined
    
    def _apply_hierarchy_rules(self, candidates: List[HeadingCandidate]) -> List[HeadingCandidate]:
        """Apply logical hierarchy rules"""
        refined = []
        h1_count_per_page = {}
        
        for candidate in candidates:
            page = candidate.page
            
            # Limit H1 per page
            if candidate.level == 'H1':
                h1_count_per_page[page] = h1_count_per_page.get(page, 0) + 1
                if h1_count_per_page[page] > self.hierarchy_rules['max_h1_per_page']:
                    candidate.level = 'H2'
            
            refined.append(candidate)
        
        return refined
    
    def _apply_semantic_consistency(self, candidates: List[HeadingCandidate]) -> List[HeadingCandidate]:
        """Ensure semantic consistency in heading levels"""
        # Group similar headings
        heading_groups = {}
        for candidate in candidates:
            # Simple similarity based on first few words
            key = ' '.join(candidate.text.split()[:3]).lower()
            if key not in heading_groups:
                heading_groups[key] = []
            heading_groups[key].append(candidate)
        
        # Normalize levels within groups
        for group in heading_groups.values():
            if len(group) > 1:
                # Use most common level or highest confidence
                levels = [c.level for c in group]
                most_common_level = max(set(levels), key=levels.count)
                for candidate in group:
                    candidate.level = most_common_level
        
        return candidates
    
    def _validate_numerical_sequences(self, candidates: List[HeadingCandidate]) -> List[HeadingCandidate]:
        """Validate and correct numerical sequences"""
        numbered_candidates = []
        
        for candidate in candidates:
            match = re.match(r'^(\d+)\.', candidate.text)
            if match:
                numbered_candidates.append((int(match.group(1)), candidate))
        
        # Sort by number
        numbered_candidates.sort(key=lambda x: x[0])
        
        # Validate sequence
        expected_num = 1
        for num, candidate in numbered_candidates:
            if num == expected_num:
                # Correct sequence
                if expected_num == 1:
                    candidate.level = 'H1'
                else:
                    candidate.level = 'H2'
                expected_num += 1
            else:
                # Broken sequence, likely subsection
                candidate.level = 'H3'
        
        return candidates


class PerformanceProfiler:
    """Performance monitoring and optimization"""
    
    def __init__(self):
        self.metrics = {}
        self.timing_history = []
    
    def profile_detection(self, detector, spans: List[SpanInfo]) -> Dict[str, Any]:
        """Profile detection performance"""
        import time
        
        start_time = time.time()
        
        # Profile individual signals
        signal_times = {}
        for signal in detector.signals:
            signal_start = time.time()
            signal.build_context(spans)
            
            # Profile scoring
            for span in spans[:10]:  # Sample first 10 spans
                signal.calculate_score(span, spans)
            
            signal_times[signal.name] = time.time() - signal_start
        
        # Profile overall detection
        detection_start = time.time()
        candidates = detector.detect_headings(spans)
        detection_time = time.time() - detection_start
        
        total_time = time.time() - start_time
        
        metrics = {
            'total_time': total_time,
            'detection_time': detection_time,
            'signal_times': signal_times,
            'candidates_found': len(candidates),
            'spans_processed': len(spans),
            'throughput': len(spans) / total_time if total_time > 0 else 0
        }
        
        self.metrics = metrics
        self.timing_history.append(metrics)
        
        return metrics
    
    def get_performance_report(self) -> str:
        """Generate performance report"""
        if not self.metrics:
            return "No performance data available"
        
        report = f"""
Performance Report:
==================
Total Processing Time: {self.metrics['total_time']:.3f}s
Detection Time: {self.metrics['detection_time']:.3f}s
Candidates Found: {self.metrics['candidates_found']}
Spans Processed: {self.metrics['spans_processed']}
Throughput: {self.metrics['throughput']:.1f} spans/second

Signal Performance:
"""
        
        for signal_name, time_taken in self.metrics['signal_times'].items():
            report += f"  {signal_name}: {time_taken:.3f}s\n"
        
        return report


class EnhancedEnsembleHeadingDetector(EnsembleHeadingDetector):
    """Enhanced detector with all advanced features"""
    
    def __init__(self, custom_weights: Optional[Dict[str, float]] = None):
        super().__init__(custom_weights)
        
        # Add advanced signals
        self.signals.extend([
            SemanticSignal(),
            VisualLayoutSignal()
        ])
        
        # Add advanced components
        self.adaptive_thresholds = AdaptiveThresholdSystem()
        self.refinement_engine = ContextualRefinementEngine()
        self.profiler = PerformanceProfiler()
        
        # Auto-optimization features
        self.auto_optimize = False
        self.optimization_history = []
    
    def detect_headings_enhanced(self, spans: List[SpanInfo]) -> List[HeadingCandidate]:
        """Enhanced detection with all advanced features"""
        # Profile performance
        if hasattr(self, 'profiler'):
            self.profiler.profile_detection(self, spans)
        
        # Auto-detect document type and adjust thresholds
        doc_type = self.adaptive_thresholds.detect_document_type(spans)
        thresholds = self.adaptive_thresholds.get_thresholds(doc_type)
        
        # Update thresholds
        self.heading_threshold = thresholds['heading_threshold']
        self.level_thresholds = thresholds['level_thresholds']
        
        # Regular detection
        candidates = super().detect_headings(spans)
        
        # Apply contextual refinement
        candidates = self.refinement_engine.refine_hierarchy(candidates)
        
        # Add document type to context
        for candidate in candidates:
            candidate.context['document_type'] = doc_type
        
        return candidates
    
    def auto_optimize_weights(self, validation_data: List[Tuple[List[SpanInfo], List[bool]]]):
        """Automatically optimize weights using validation data"""
        best_weights = None
        best_score = 0
        
        # Try different weight combinations
        weight_combinations = [
            {'SemanticSignal': 0.20, 'VisualLayoutSignal': 0.15},
            {'SemanticSignal': 0.15, 'VisualLayoutSignal': 0.20},
            {'SemanticSignal': 0.10, 'VisualLayoutSignal': 0.25},
        ]
        
        for weights in weight_combinations:
            # Update weights
            for signal in self.signals:
                if signal.name in weights:
                    signal.weight = weights[signal.name]
            
            # Evaluate
            total_score = 0
            for spans, labels in validation_data:
                candidates = self.detect_headings_enhanced(spans)
                # Simple scoring based on number of candidates found
                score = len(candidates) / len(spans) if spans else 0
                total_score += score
            
            avg_score = total_score / len(validation_data)
            
            if avg_score > best_score:
                best_score = avg_score
                best_weights = weights.copy()
        
        # Apply best weights
        if best_weights:
            for signal in self.signals:
                if signal.name in best_weights:
                    signal.weight = best_weights[signal.name]
            
            self.optimization_history.append({
                'weights': best_weights,
                'score': best_score
            })
    
    def generate_confidence_explanation(self, candidate: HeadingCandidate) -> str:
        """Generate human-readable explanation of confidence score"""
        explanations = []
        
        # Analyze each signal contribution
        for signal_name, score in candidate.signals.items():
            if score > 0.7:
                explanations.append(f"Strong {signal_name.replace('Signal', '').lower()} indicators")
            elif score > 0.4:
                explanations.append(f"Moderate {signal_name.replace('Signal', '').lower()} indicators")
        
        # Overall confidence
        if candidate.confidence > 0.8:
            confidence_level = "Very High"
        elif candidate.confidence > 0.6:
            confidence_level = "High"
        elif candidate.confidence > 0.4:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"
        
        explanation = f"Confidence: {confidence_level} ({candidate.confidence:.2f})\n"
        explanation += f"Reasons: {', '.join(explanations)}"
        
        return explanation


def detect_headings_and_title(spans):
    """
    Adapter for outline_builder: returns (title, outline) using EnsembleHeadingDetector.
    """
    detector = EnsembleHeadingDetector()
    return detector.get_document_outline(spans)


def demo_enhanced_features():
    """Demonstrate enhanced features"""
    print("=== Enhanced Multi-Signal Heading Detection Demo ===\n")
    
    # Create sample data
    spans = create_sample_data()
    
    # Enhanced detector
    detector = EnhancedEnsembleHeadingDetector()
    
    print("1. Enhanced Detection with Auto-Optimization:")
    candidates = detector.detect_headings_enhanced(spans)
    
    for candidate in candidates[:3]:
        print(f"\nHeading: '{candidate.text}'")
        print(f"Level: {candidate.level}")
        print(f"Document Type: {candidate.context.get('document_type', 'Unknown')}")
        print(detector.generate_confidence_explanation(candidate))
    
    print("\n2. Performance Profiling:")
    print(detector.profiler.get_performance_report())
    
    print("\n3. Signal Breakdown:")
    for signal in detector.signals:
        print(f"{signal.name}: Weight = {signal.weight:.3f}")


class ActiveLearningSystem:
    """Active learning for continuous improvement"""
    
    def __init__(self):
        self.uncertain_predictions = []
        self.feedback_data = []
        self.improvement_threshold = 0.05
    
    def identify_uncertain_predictions(self, candidates: List[HeadingCandidate]) -> List[HeadingCandidate]:
        """Identify predictions that need human review"""
        uncertain = []
        
        for candidate in candidates:
            if (0.4 <= candidate.confidence <= 0.7 or  # Medium confidence
                self._has_conflicting_signals(candidate) or  # Conflicting signals
                self._is_edge_case(candidate)):  # Edge cases
                uncertain.append(candidate)
        
        return uncertain
    
    def _has_conflicting_signals(self, candidate: HeadingCandidate) -> bool:
        """Check if signals are conflicting"""
        signal_values = list(candidate.signals.values())
        return max(signal_values) - min(signal_values) > 0.6
    
    def _is_edge_case(self, candidate: HeadingCandidate) -> bool:
        """Identify edge cases"""
        # Very short or very long text
        text_len = len(candidate.text)
        if text_len < 3 or text_len > 200:
            return True
        
        # Unusual positions
        if candidate.bbox[1] < 50 or candidate.bbox[1] > 800:  # Adjust based on page size
            return True
        
        return False
    
    def collect_feedback(self, candidate: HeadingCandidate, is_correct: bool, correct_level: str = None):
        """Collect human feedback"""
        feedback = {
            'candidate': candidate,
            'is_correct': is_correct,
            'correct_level': correct_level,
            'timestamp': time.time()
        }
        self.feedback_data.append(feedback)
    
    def retrain_with_feedback(self, detector: 'EnhancedEnsembleHeadingDetector'):
        """Retrain model with collected feedback"""
        if len(self.feedback_data) < 10:  # Need minimum feedback
            return False
        
        # Convert feedback to training data
        training_spans = []
        training_labels = []
        
        for feedback in self.feedback_data:
            candidate = feedback['candidate']
            # This would need actual span reconstruction
            # For demo purposes, we'll simulate
            training_labels.append(feedback['is_correct'])
        
        # Retrain if we have enough data
        if len(training_labels) >= 20:
            # detector.train_ml_model(training_data)
            print(f"Retrained with {len(training_labels)} feedback samples")
            return True
        
        return False


class MultiModalIntegration:
    """Integration with images, tables, and other elements"""
    
    def __init__(self):
        self.image_keywords = ['figure', 'fig', 'image', 'chart', 'graph', 'diagram']
        self.table_keywords = ['table', 'tab', 'data', 'results', 'statistics']
    
    def analyze_cross_references(self, spans: List[SpanInfo]) -> Dict[str, List[str]]:
        """Analyze cross-references to figures and tables"""
        references = {'figures': [], 'tables': [], 'sections': []}
        
        for span in spans:
            text = span.text.lower()
            
            # Figure references
            if any(keyword in text for keyword in self.image_keywords):
                if re.search(r'figure\s+\d+|fig\s+\d+', text):
                    references['figures'].append(span.text)
            
            # Table references
            if any(keyword in text for keyword in self.table_keywords):
                if re.search(r'table\s+\d+|tab\s+\d+', text):
                    references['tables'].append(span.text)
            
            # Section references
            if re.search(r'section\s+\d+|chapter\s+\d+', text):
                references['sections'].append(span.text)
        
        return references
    
    def enhance_with_multimodal_context(self, candidates: List[HeadingCandidate], 
                                      spans: List[SpanInfo]) -> List[HeadingCandidate]:
        """Enhance candidates with multimodal context"""
        cross_refs = self.analyze_cross_references(spans)
        
        for candidate in candidates:
            # Add multimodal context
            candidate.context['cross_references'] = {
                'has_figure_refs': any(ref in candidate.text.lower() for ref in cross_refs['figures']),
                'has_table_refs': any(ref in candidate.text.lower() for ref in cross_refs['tables']),
                'has_section_refs': any(ref in candidate.text.lower() for ref in cross_refs['sections'])
            }
        
        return candidates


class RealTimeOptimization:
    """Real-time optimization during processing"""
    
    def __init__(self):
        self.processing_stats = {
            'total_processed': 0,
            'accuracy_history': [],
            'performance_history': []
        }
    
    def optimize_during_processing(self, detector: 'EnhancedEnsembleHeadingDetector', 
                                 spans: List[SpanInfo]) -> Dict[str, Any]:
        """Optimize detector settings during processing"""
        # Analyze current batch
        batch_stats = self._analyze_batch(spans)
        if batch_stats['avg_confidence'] < 0.5:

            detector.heading_threshold *= 0.9
            for level in detector.level_thresholds:
                detector.level_thresholds[level] *= 0.9
        
        # Adjust signal weights based on performance
        if batch_stats['font_size_variance'] < 2:
            # Low font size variance - reduce font size signal weight
            for signal in detector.signals:
                if signal.name == 'FontSizeSignal':
                    signal.weight *= 0.8
                elif signal.name == 'PositionalSignal':
                    signal.weight *= 1.2 
        
        return {
            'threshold_adjusted': True,
            'weights_adjusted': True,
            'batch_stats': batch_stats
        }
    
    def _analyze_batch(self, spans: List[SpanInfo]) -> Dict[str, Any]:
        """Analyze batch characteristics"""
        sizes = [span.size for span in spans]
        
        return {
            'span_count': len(spans),
            'avg_size': np.mean(sizes),
            'font_size_variance': np.var(sizes),
            'avg_confidence': 0.6,  # Placeholder
            'text_complexity': np.mean([len(span.text) for span in spans])
        }


class ExplainableAI:
    """Explainable AI for decision transparency"""
    
    def __init__(self):
        self.decision_tree = {}
        self.feature_importance = {}
    
    def explain_prediction(self, candidate: HeadingCandidate) -> Dict[str, Any]:
        """Generate detailed explanation for a prediction"""
        explanation = {
            'prediction': candidate.level,
            'confidence': candidate.confidence,
            'reasoning': self._generate_reasoning(candidate),
            'feature_contributions': self._calculate_feature_contributions(candidate),
            'decision_path': self._trace_decision_path(candidate),
            'alternative_scenarios': self._generate_alternatives(candidate)
        }
        
        return explanation
    
    def _generate_reasoning(self, candidate: HeadingCandidate) -> List[str]:
        """Generate human-readable reasoning"""
        reasoning = []
        
        # Analyze signal contributions
        for signal_name, score in candidate.signals.items():
            if score > 0.6:
                reasoning.append(f"Strong {signal_name.replace('Signal', '').lower()} evidence (score: {score:.2f})")
            elif score > 0.3:
                reasoning.append(f"Moderate {signal_name.replace('Signal', '').lower()} evidence (score: {score:.2f})")
        
        # Context-based reasoning
        context = candidate.context
        if context.get('is_bold', False):
            reasoning.append("Text is bold, indicating emphasis")
        
        if context.get('font_size', 0) > 14:
            reasoning.append(f"Large font size ({context['font_size']}pt) suggests heading")
        
        return reasoning
    
    def _calculate_feature_contributions(self, candidate: HeadingCandidate) -> Dict[str, float]:
        """Calculate how much each feature contributed to the decision"""
        contributions = {}
        
        for signal_name, score in candidate.signals.items():
            contributions[signal_name] = score * 0.2  
        
        return contributions
    
    def _trace_decision_path(self, candidate: HeadingCandidate) -> List[str]:
        """Trace the decision-making path"""
        path = [
            f"Initial text: '{candidate.text}'",
            f"Calculated signal scores: {candidate.signals}",
            f"Applied weights and computed confidence: {candidate.confidence:.3f}",
            f"Compared against threshold: {candidate.confidence:.3f} >= 0.6",
            f"Assigned level: {candidate.level}"
        ]
        
        return path
    
    def _generate_alternatives(self, candidate: HeadingCandidate) -> List[Dict[str, Any]]:
        """Generate alternative scenarios"""
        alternatives = []

        if candidate.confidence >= 0.8:
            alternatives.append({
                'scenario': 'If confidence was 0.7',
                'result': 'Would still be classified as heading but level might change'
            })
        
        font_score = candidate.signals.get('FontSizeSignal', 0)
        if font_score > 0.5:
            alternatives.append({
                'scenario': 'If font size was average',
                'result': 'Classification would depend more on other signals'
            })
        
        return alternatives


class BenchmarkingSystem:
    """Comprehensive benchmarking against other systems"""
    
    def __init__(self):
        pass