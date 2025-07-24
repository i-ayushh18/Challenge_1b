import sys
import logging
import os

# Determine log directory based on --debug_output argument if present
log_dir = None
for i, arg in enumerate(sys.argv):
    if arg == '--debug_output' and i + 1 < len(sys.argv):
        log_dir = os.path.dirname(sys.argv[i + 1])
        break
if not log_dir:
    log_dir = os.environ.get("LOG_DIR", "debug_output")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "pipeline.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(name)s: %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Union
import json
from datetime import datetime
from enum import Enum
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from collections import Counter
# Text processing utilities
from sklearn.metrics.pairwise import cosine_similarity
import itertools
# Remove all transformer/torch/model code

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import TruncatedSVD
import nltk
# nltk.download('punkt', quiet=True)  # Removed for offline compliance; punkt is installed in Docker build
from nltk.tokenize import sent_tokenize
from rank_bm25 import BM25Okapi
import pytesseract
from PIL import Image
import spacy
nlp = spacy.load('en_core_web_sm')


import nltk
nltk.data.path.append('/usr/local/nltk_data')

def extract_dynamic_personas_jobs(document_text, n_clusters=3):
    sentences = sent_tokenize(document_text)
    if len(sentences) < n_clusters:
        sorted_sents = sorted(sentences, key=len, reverse=True)
        return sorted_sents[:n_clusters], sorted_sents[:n_clusters]

    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    try:
        X = vectorizer.fit_transform(sentences)
        
        # Dimensionality reduction for better clustering
        if X.shape[1] > 100:  # Only use SVD if we have many features
            svd = TruncatedSVD(n_components=min(100, X.shape[1]-1))
            X_reduced = svd.fit_transform(X)
        else:
            X_reduced = X.toarray()
            
        # Cluster using KMeans
        kmeans = KMeans(n_clusters=min(n_clusters, len(sentences)), random_state=42, n_init=10)
        kmeans.fit(X_reduced)
        
        persona_candidates = []
        job_candidates = []
        
        for cluster_id in range(kmeans.n_clusters):
            cluster_indices = np.where(kmeans.labels_ == cluster_id)[0]
            if len(cluster_indices) == 0:
                continue
                
            centroid = kmeans.cluster_centers_[cluster_id]
            cluster_points = X_reduced[cluster_indices]
            
            # Calculate distances to centroid
            dists = np.linalg.norm(cluster_points - centroid, axis=1)
            closest_in_cluster_idx = np.argmin(dists)
            closest_sentence_idx = cluster_indices[closest_in_cluster_idx]
            candidate_sentence = sentences[closest_sentence_idx]
            
            # Using spaCy to extract persona (role/entity) and job (action/verb)
            doc = nlp(candidate_sentence)
            
            # Extract persona (noun phrases or named entities)
            persona_candidate = None
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG", "NORP", "TITLE", "PROFESSION"] or \
                   any(word in ent.text.lower() for word in ["engineer", "manager", "scientist", "analyst"]):
                    persona_candidate = ent.text
                    break
                    
            if not persona_candidate:
                for chunk in doc.noun_chunks:
                    persona_candidate = chunk.text
                    break
                    
            # Extract job (verb phrases or actions)
            job_candidate = None
            for token in doc:
                if token.pos_ == "VERB" and token.dep_ != "aux":
                    # Get the full verb phrase
                    job_candidate = token.text
                    for child in token.children:
                        if child.dep_ in ("dobj", "attr", "prep"):
                            job_candidate += " " + child.text
                    break
                    
            if persona_candidate:
                persona_candidates.append(persona_candidate)
            if job_candidate:
                job_candidates.append(job_candidate)
                
        # Fallback if no personas/jobs found
        if not persona_candidates or not job_candidates:
            return sentences[:n_clusters], sentences[:n_clusters]
            
        return persona_candidates[:n_clusters], job_candidates[:n_clusters]
        
    except Exception as e:
        logger.warning(f"Error in extract_dynamic_personas_jobs: {str(e)}")
        # Fallback to simple sentence selection
        return sentences[:n_clusters], sentences[:n_clusters]
    # Fallback if nothing found
    if not persona_candidates:
        persona_candidates = [sentences[0]]
    if not job_candidates:
        job_candidates = [sentences[0]]
    return persona_candidates, job_candidates

# Example candidate personas and jobs
persona_descriptions = [
    "Technical professional (developer, engineer, IT administrator)",
    "Business decision maker (executive, manager, investor)",
    "Research professional (academic, scientist, analyst)",
    "Legal professional (lawyer, compliance officer, legal counsel)",
    "Marketing professional (marketer, brand manager, campaign manager)",
    "Financial professional (analyst, accountant, investor)",
    "Product user (end user, customer, support staff)",
    "Policy implementer (manager, supervisor, compliance officer)",
    "Project manager (project lead, scrum master, coordinator)",
    "Human resources professional (HR manager, recruiter, talent acquisition)",
    "Sales professional (sales manager, account executive, business development)",
    "Customer support specialist (customer service, help desk, support agent)",
    "Operations manager (operations lead, logistics, supply chain)",
    "Quality assurance specialist (QA engineer, tester, auditor)",
    "Data scientist (data analyst, machine learning engineer, statistician)",
    "Creative professional (designer, writer, artist, content creator)",
    "Educator (teacher, trainer, instructional designer, professor)",
    "Healthcare professional (doctor, nurse, medical researcher)",
    "IT security specialist (security analyst, cybersecurity expert)",
    "Procurement specialist (buyer, sourcing manager, vendor manager)",
    "Facilities manager (workplace safety, building operations)",
    "Event planner (event manager, conference coordinator)",
    "Communications professional (PR, internal comms, spokesperson)",
    "Product manager (product owner, product strategist)",
    "Entrepreneur (startup founder, small business owner)",
    "Consultant (management consultant, advisor, strategist)",
    "Government official (policy maker, regulator, public servant)",
    "Nonprofit leader (NGO director, program manager)",
    "Investor (venture capitalist, angel investor, portfolio manager)",
    "Legal assistant (paralegal, legal secretary)",
    "Media professional (journalist, editor, broadcaster)",
    "Supply chain analyst (logistics coordinator, inventory manager)",
    "Environmental specialist (sustainability officer, environmental engineer)",
    "Real estate professional (broker, property manager)",
    "Finance executive (CFO, controller, treasurer)",
    "Administrative assistant (office manager, executive assistant)",
    "Trainer (corporate trainer, learning & development)",
    "UX researcher (user researcher, usability analyst)",
    "Mobile app developer (iOS developer, Android developer)",
    "Cloud architect (cloud engineer, DevOps architect)"
]

job_descriptions = [
    "Implement, configure, or troubleshoot technical solutions and systems",
    "Evaluate business opportunities and make strategic decisions",
    "Understand research findings and apply insights to work",
    "Review legal documents and ensure compliance",
    "Create effective marketing strategies and campaigns",
    "Analyze financial performance and make investment decisions",
    "Learn how to use products effectively and solve problems",
    "Implement and enforce organizational policies and procedures",
    "Plan and execute successful product launches",
    "Develop and manage project timelines and deliverables",
    "Optimize supply chain operations for efficiency",
    "Conduct market research to identify new opportunities",
    "Design and deliver engaging training programs",
    "Manage customer relationships and resolve issues",
    "Ensure data privacy and security compliance",
    "Drive digital transformation initiatives",
    "Coordinate cross-functional teams for project success",
    "Develop and monitor key performance indicators (KPIs)",
    "Prepare and present executive-level reports",
    "Lead organizational change management efforts",
    "Develop and implement sustainability strategies",
    "Oversee vendor and partner relationships",
    "Manage budgets and control costs",
    "Develop and execute crisis communication plans",
    "Enhance employee engagement and workplace culture",
    "Develop innovative product features based on user feedback",
    "Create technical documentation for end users",
    "Facilitate workshops and brainstorming sessions",
    "Negotiate contracts and agreements with stakeholders",
    "Develop and implement risk management strategies",
    "Monitor and improve customer satisfaction metrics",
    "Lead software development sprints and agile ceremonies",
    "Develop and execute go-to-market strategies",
    "Analyze and visualize large datasets for insights",
    "Develop and maintain cloud infrastructure",
    "Manage regulatory compliance and audit processes",
    "Develop and implement quality assurance protocols",
    "Plan and manage corporate events and conferences",
    "Develop and execute internal communication strategies",
    "Mentor and coach team members for professional growth",
    "Develop and implement onboarding programs for new hires",
    "Manage intellectual property and patent filings",
    "Develop and implement social media campaigns",
    "Oversee product lifecycle management",
    "Develop and implement customer loyalty programs",
    "Manage mergers, acquisitions, and integrations",
    "Develop and implement remote work policies",
    "Oversee facility management and workplace safety",
    "Develop and implement diversity and inclusion initiatives",
    "Manage international expansion and localization efforts"
]



from outline_builder import OutlineBuilderFactory, DocumentOutline, HeadingInfo
import pdf_loader


logger = logging.getLogger(__name__)

# ==================== Enhanced Generic Document Analysis System ====================

@dataclass
class DocumentSection:
    """Represents a document section with metadata"""
    text: str
    document_name: str
    page: int
    heading_info: HeadingInfo
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    refined_text: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'text': self.text,
            'document_name': self.document_name,
            'page': self.page,
            'confidence': self.confidence,
            'metadata': self.metadata,
            'refined_text': self.refined_text
        }

class AdaptiveDocumentTypePredictor:
    """Advanced, generic document type predictor using multiple analysis approaches"""
    
    def __init__(self):
        # Enhanced job title patterns for better persona matching
        self.job_title_patterns = {
            'executive': {
                'titles': ['CEO', 'CTO', 'CFO', 'COO', 'VP', 'Vice President', 'Director', 'Head of', 'Chief', 'President', 'Executive', 'Senior Director', 'Managing Director'],
                'contexts': ['board', 'stakeholder', 'executive team', 'leadership', 'strategy', 'vision', 'governance'],
                'responsibilities': ['strategic', 'leadership', 'oversight', 'decision making', 'planning']
            },
            'technical': {
                'titles': ['Engineer', 'Developer', 'Architect', 'Programmer', 'Analyst', 'Specialist', 'Administrator', 'Technician', 'Consultant', 'Lead Developer', 'Senior Engineer', 'Principal Engineer'],
                'contexts': ['development', 'programming', 'coding', 'technical', 'system', 'software', 'hardware', 'infrastructure'],
                'responsibilities': ['implement', 'develop', 'code', 'design', 'troubleshoot', 'maintain', 'optimize']
            },
            'creative': {
                'titles': ['Designer', 'Artist', 'Creative', 'UX', 'UI', 'Writer', 'Content Creator', 'Brand Manager', 'Creative Director', 'Art Director'],
                'contexts': ['design', 'creative', 'visual', 'aesthetic', 'brand', 'content', 'user experience'],
                'responsibilities': ['design', 'create', 'conceptualize', 'visualize', 'brand', 'communicate']
            },
            'business': {
                'titles': ['Manager', 'Coordinator', 'Supervisor', 'Lead', 'Business Analyst', 'Product Manager', 'Project Manager', 'Operations Manager'],
                'contexts': ['business', 'management', 'operations', 'coordination', 'planning', 'execution'],
                'responsibilities': ['manage', 'coordinate', 'plan', 'execute', 'analyze', 'optimize', 'oversee']
            },
            'sales_marketing': {
                'titles': ['Sales', 'Marketing', 'Account Manager', 'Business Development', 'Marketing Manager', 'Sales Representative', 'Account Executive'],
                'contexts': ['sales', 'marketing', 'customer', 'client', 'revenue', 'growth', 'campaigns'],
                'responsibilities': ['sell', 'market', 'promote', 'engage', 'convert', 'grow', 'acquire']
            },
            'research': {
                'titles': ['Researcher', 'Scientist', 'Analyst', 'Data Scientist', 'Research Analyst', 'Principal Researcher'],
                'contexts': ['research', 'analysis', 'data', 'study', 'investigation', 'findings', 'insights'],
                'responsibilities': ['research', 'analyze', 'investigate', 'study', 'discover', 'evaluate']
            }
        }
        
        # Industry-specific keyword libraries
        self.industry_keywords = {
            'technology': ['software', 'hardware', 'cloud', 'AI', 'machine learning', 'data science', 'cybersecurity', 'blockchain', 'IoT', 'API', 'SDK', 'DevOps', 'agile', 'scrum'],
            'finance': ['financial', 'banking', 'investment', 'portfolio', 'risk management', 'compliance', 'audit', 'accounting', 'fintech', 'trading', 'assets', 'liabilities'],
            'healthcare': ['medical', 'healthcare', 'patient', 'clinical', 'pharmaceutical', 'diagnosis', 'treatment', 'therapy', 'medical device', 'health records'],
            'education': ['education', 'learning', 'curriculum', 'academic', 'student', 'teacher', 'training', 'course', 'university', 'school', 'pedagogy'],
            'retail': ['retail', 'e-commerce', 'customer', 'shopping', 'merchandise', 'inventory', 'supply chain', 'point of sale', 'consumer'],
            'manufacturing': ['manufacturing', 'production', 'quality control', 'supply chain', 'logistics', 'operations', 'assembly', 'factory', 'industrial'],
            'consulting': ['consulting', 'advisory', 'strategy', 'transformation', 'optimization', 'best practices', 'implementation', 'change management'],
            'media': ['media', 'content', 'publishing', 'journalism', 'broadcasting', 'digital media', 'social media', 'communications']
        }
        
        # Experience level patterns
        self.experience_patterns = {
            'entry_level': {
                'indicators': ['entry level', 'junior', 'graduate', 'intern', 'trainee', '0-2 years', 'new graduate', 'recent graduate'],
                'contexts': ['learning', 'training', 'mentorship', 'guidance', 'development', 'onboarding']
            },
            'mid_level': {
                'indicators': ['mid-level', 'experienced', 'specialist', '3-5 years', '2-4 years', 'professional'],
                'contexts': ['independent', 'responsible', 'project', 'team member', 'contributor']
            },
            'senior_level': {
                'indicators': ['senior', 'lead', 'principal', '5+ years', '6+ years', 'expert', 'advanced'],
                'contexts': ['leadership', 'mentoring', 'complex', 'strategic', 'guidance', 'expertise']
            },
            'executive_level': {
                'indicators': ['executive', 'director', 'VP', 'C-level', '10+ years', 'executive team'],
                'contexts': ['strategic', 'organizational', 'high-level', 'board', 'stakeholder', 'vision']
            }
        }
        
        # Core document type patterns with enhanced generic coverage
        self.document_patterns = {
            'technical_manual': {
                'keywords': ['installation', 'configuration', 'setup', 'troubleshooting', 'api', 'sdk', 'integration', 'deployment', 'maintenance', 'technical', 'code', 'function', 'parameter', 'endpoint', 'protocol', 'specification'],
                'patterns': [r'how to', r'step by step', r'configuration', r'installation guide', r'user manual', r'technical documentation', r'api reference', r'getting started'],
                'semantic_themes': ['technical', 'implementation', 'development', 'engineering', 'system'],
                'persona': 'Technical professional (developer, engineer, IT administrator, system analyst)',
                'job_to_be_done': 'Implement, configure, or troubleshoot technical solutions and systems'
            },
            'business_proposal': {
                'keywords': ['proposal', 'business case', 'roi', 'investment', 'market analysis', 'financial', 'strategy', 'partnership', 'executive', 'stakeholder', 'opportunity', 'growth', 'revenue', 'profitability'],
                'patterns': [r'business proposal', r'executive summary', r'financial analysis', r'market opportunity', r'strategic plan', r'investment case'],
                'semantic_themes': ['business', 'strategy', 'financial', 'market', 'investment'],
                'persona': 'Business decision maker (executive, manager, investor, strategist)',
                'job_to_be_done': 'Evaluate business opportunities and make strategic decisions'
            },
            'research_paper': {
                'keywords': ['research', 'study', 'methodology', 'results', 'conclusion', 'hypothesis', 'data analysis', 'literature review', 'experiment', 'findings', 'academic', 'scientific', 'investigation'],
                'patterns': [r'abstract', r'introduction', r'methodology', r'results', r'conclusion', r'references', r'literature review', r'discussion'],
                'semantic_themes': ['research', 'analysis', 'investigation', 'study', 'academic'],
                'persona': 'Research professional (academic, scientist, analyst, researcher)',
                'job_to_be_done': 'Understand research findings and apply insights to work'
            },
            'legal_document': {
                'keywords': ['legal', 'contract', 'agreement', 'terms', 'conditions', 'liability', 'compliance', 'regulatory', 'law', 'clause', 'jurisdiction', 'governance', 'policy', 'regulation'],
                'patterns': [r'terms and conditions', r'agreement', r'contract', r'legal notice', r'compliance', r'governing law', r'disclaimer'],
                'semantic_themes': ['legal', 'compliance', 'regulatory', 'governance', 'contractual'],
                'persona': 'Legal professional (lawyer, compliance officer, legal counsel, regulatory specialist)',
                'job_to_be_done': 'Review legal documents and ensure compliance'
            },
            'marketing_material': {
                'keywords': ['marketing', 'campaign', 'brand', 'customer', 'product', 'features', 'benefits', 'target audience', 'conversion', 'advertising', 'promotion', 'sales', 'market', 'competitive'],
                'patterns': [r'marketing campaign', r'product features', r'customer benefits', r'brand strategy', r'target market', r'competitive analysis'],
                'semantic_themes': ['marketing', 'brand', 'customer', 'promotion', 'sales'],
                'persona': 'Marketing professional (marketer, brand manager, campaign manager, sales professional)',
                'job_to_be_done': 'Create effective marketing strategies and campaigns'
            },
            'financial_report': {
                'keywords': ['financial', 'revenue', 'profit', 'loss', 'balance sheet', 'income statement', 'cash flow', 'quarterly', 'annual', 'earnings', 'assets', 'liabilities', 'equity', 'audit'],
                'patterns': [r'financial report', r'quarterly results', r'annual report', r'balance sheet', r'income statement', r'cash flow statement', r'financial statements'],
                'semantic_themes': ['financial', 'accounting', 'revenue', 'profit', 'audit'],
                'persona': 'Financial professional (analyst, accountant, investor, auditor)',
                'job_to_be_done': 'Analyze financial performance and make investment decisions'
            },
            'product_documentation': {
                'keywords': ['product', 'features', 'specifications', 'user guide', 'documentation', 'help', 'support', 'tutorial', 'manual', 'instructions', 'how to use', 'product guide'],
                'patterns': [r'product documentation', r'user guide', r'features', r'specifications', r'help', r'getting started', r'user manual'],
                'semantic_themes': ['product', 'user', 'guide', 'instruction', 'support'],
                'persona': 'Product user (end user, customer, support staff, product manager)',
                'job_to_be_done': 'Learn how to use products effectively and solve problems'
            },
            'policy_document': {
                'keywords': ['policy', 'procedure', 'guidelines', 'standards', 'compliance', 'rules', 'regulations', 'governance', 'framework', 'protocol', 'best practices', 'operational'],
                'patterns': [r'policy', r'procedure', r'guidelines', r'standards', r'compliance', r'operational manual', r'best practices'],
                'semantic_themes': ['policy', 'procedure', 'governance', 'compliance', 'operational'],
                'persona': 'Policy implementer (manager, supervisor, compliance officer, operations specialist)',
                'job_to_be_done': 'Implement and enforce organizational policies and procedures'
            },
            'educational_material': {
                'keywords': ['education', 'learning', 'course', 'curriculum', 'training', 'instruction', 'lesson', 'academic', 'teaching', 'student', 'knowledge', 'skills', 'competency'],
                'patterns': [r'course outline', r'learning objectives', r'curriculum', r'lesson plan', r'educational material', r'training guide'],
                'semantic_themes': ['education', 'learning', 'teaching', 'academic', 'knowledge'],
                'persona': 'Educational professional (teacher, trainer, instructional designer, student)',
                'job_to_be_done': 'Learn new skills and acquire knowledge effectively'
            },
            'creative_content': {
                'keywords': ['creative', 'design', 'art', 'content', 'story', 'narrative', 'creative writing', 'visual', 'aesthetic', 'branding', 'communication', 'expression'],
                'patterns': [r'creative brief', r'design guide', r'brand guidelines', r'content strategy', r'creative direction'],
                'semantic_themes': ['creative', 'design', 'art', 'content', 'expression'],
                'persona': 'Creative professional (designer, writer, artist, content creator)',
                'job_to_be_done': 'Create compelling creative content and designs'
            }
        }
        
        # Generic persona patterns for unknown document types
        self.generic_personas = {
            'analytical': 'Analytical professional (analyst, researcher, data scientist)',
            'operational': 'Operational professional (manager, supervisor, coordinator)',
            'strategic': 'Strategic professional (executive, strategist, planner)',
            'technical': 'Technical professional (engineer, developer, specialist)',
            'creative': 'Creative professional (designer, writer, artist)',
            'customer_facing': 'Customer-facing professional (sales, support, service)'
        }
        
        # Generic jobs-to-be-done patterns
        self.generic_jobs = {
            'analytical': 'Analyze information and derive insights for decision making',
            'operational': 'Execute tasks and manage day-to-day operations effectively',
            'strategic': 'Develop strategies and make high-level decisions',
            'technical': 'Implement technical solutions and solve complex problems',
            'creative': 'Create compelling content and innovative solutions',
            'customer_facing': 'Serve customers and build relationships'
        }
    
    def predict_document_type(self, document_content: str, headings: List[str]) -> Dict[str, Any]:
        """Enhanced document type prediction using multiple analysis approaches, now with n-gram and bigram support"""
        content_lower = document_content.lower()
        headings_lower = [h.lower() for h in headings]
        
        # Multi-factor scoring system
        scores = {}
        confidence_factors = {}
        
        for doc_type, patterns in self.document_patterns.items():
            score = 0
            confidence_indicators = []
            
            # 1. Keyword analysis (weighted by frequency and position, now with n-gram/bigram support)
            keyword_score = 0
            keywords = patterns['keywords']
            # Generate bigrams from keywords
            bigrams = [' '.join(pair) for pair in zip(keywords, keywords[1:])]
            all_ngrams = keywords + bigrams
            # Check n-grams in content
            for ngram in all_ngrams:
                ngram_lower = ngram.lower()
                if ngram_lower in content_lower:
                    # Bigram gets higher score
                    if ' ' in ngram_lower:
                        keyword_score += 4
                        keyword_score += content_lower.count(ngram_lower) * 1.0
                    else:
                        keyword_score += 2
                        keyword_score += content_lower.count(ngram_lower) * 0.5
                # Check in headings (higher weight)
                for heading in headings_lower:
                    if ngram_lower in heading:
                        if ' ' in ngram_lower:
                            keyword_score += 8
                            confidence_indicators.append(f"bigram_in_heading:{ngram}")
                        else:
                            keyword_score += 4
                            confidence_indicators.append(f"keyword_in_heading:{ngram}")
            
            # 2. Pattern matching (regex-based)
            pattern_score = 0
            for pattern in patterns['patterns']:
                if re.search(pattern, content_lower):
                    pattern_score += 3
                    confidence_indicators.append(f"pattern_match:{pattern}")
                for heading in headings_lower:
                    if re.search(pattern, heading):
                        pattern_score += 5
                        confidence_indicators.append(f"pattern_in_heading:{pattern}")
            
            # 3. Semantic theme analysis
            theme_score = 0
            for theme in patterns['semantic_themes']:
                theme_lower = theme.lower()
                if theme_lower in content_lower:
                    theme_score += 1.5
                if any(theme_lower in heading for heading in headings_lower):
                    theme_score += 2.5
            
            # 4. Document structure analysis
            structure_score = self._analyze_document_structure(headings_lower, doc_type)
            
            # 5. Content length and complexity analysis
            complexity_score = self._analyze_content_complexity(document_content, doc_type)
            
            # Combine scores with weights
            total_score = (
                keyword_score * 0.3 +
                pattern_score * 0.25 +
                theme_score * 0.2 +
                structure_score * 0.15 +
                complexity_score * 0.1
            )
            
            scores[doc_type] = total_score
            confidence_factors[doc_type] = {
                'keyword_score': keyword_score,
                'pattern_score': pattern_score,
                'theme_score': theme_score,
                'structure_score': structure_score,
                'complexity_score': complexity_score,
                'indicators': confidence_indicators
            }
        
        # Find the best match
        if scores:
            predicted_type = max(scores, key=scores.get)
            max_score = scores[predicted_type]
            
            # Calculate confidence based on score and competition
            confidence = self._calculate_confidence(max_score, scores, confidence_factors[predicted_type])
            
            if confidence >= 0.3:  # Lower threshold for more generic approach
                return {
                    'document_type': predicted_type,
                    'confidence': confidence,
                    'persona': self.document_patterns[predicted_type]['persona'],
                    'job_to_be_done': self.document_patterns[predicted_type]['job_to_be_done'],
                    'all_scores': scores,
                    'confidence_factors': confidence_factors[predicted_type]
                }
        
        # Fallback to generic prediction
        return self._generate_generic_prediction(document_content, headings_lower, scores)
    
    def _analyze_document_structure(self, headings: List[str], doc_type: str) -> float:
        """Analyze document structure patterns"""
        score = 0
        
        # Check for typical document structure patterns
        structure_patterns = {
            'technical_manual': ['introduction', 'installation', 'configuration', 'troubleshooting', 'reference'],
            'research_paper': ['abstract', 'introduction', 'methodology', 'results', 'conclusion', 'references'],
            'business_proposal': ['executive summary', 'background', 'proposal', 'financial', 'conclusion'],
            'financial_report': ['executive summary', 'financial highlights', 'balance sheet', 'income statement'],
            'policy_document': ['purpose', 'scope', 'policy', 'procedures', 'compliance']
        }
        
        if doc_type in structure_patterns:
            expected_sections = structure_patterns[doc_type]
            found_sections = sum(1 for section in expected_sections 
                               if any(section in heading for heading in headings))
            score = (found_sections / len(expected_sections)) * 5
        
        return score
    
    def _analyze_content_complexity(self, content: str, doc_type: str) -> float:
        """Analyze content complexity and technical level"""
        score = 0
        
        # Technical complexity indicators
        technical_terms = ['api', 'sdk', 'integration', 'protocol', 'algorithm', 'architecture']
        technical_count = sum(1 for term in technical_terms if term.lower() in content.lower())
        
        # Business complexity indicators
        business_terms = ['strategy', 'market', 'revenue', 'stakeholder', 'competitive', 'partnership']
        business_count = sum(1 for term in business_terms if term.lower() in content.lower())
        
        # Academic complexity indicators
        academic_terms = ['methodology', 'hypothesis', 'analysis', 'correlation', 'statistical', 'research']
        academic_count = sum(1 for term in academic_terms if term.lower() in content.lower())
        
        # Adjust score based on document type expectations
        if doc_type == 'technical_manual' and technical_count > 2:
            score += 3
        elif doc_type == 'business_proposal' and business_count > 2:
            score += 3
        elif doc_type == 'research_paper' and academic_count > 2:
            score += 3
        
        return score
    
    def _calculate_confidence(self, max_score: float, all_scores: Dict[str, float], 
                            confidence_factors: Dict[str, Any]) -> float:
        """Calculate confidence based on score strength and competition"""
        # Base confidence from score
        base_confidence = min(max_score / 10.0, 1.0)
        
        sorted_scores = sorted(all_scores.values(), reverse=True)
        if len(sorted_scores) > 1:
            score_gap = (sorted_scores[0] - sorted_scores[1]) / max(sorted_scores[0], 1)
            competition_factor = min(score_gap * 2, 1.0)
        else:
            competition_factor = 1.0
        
        # Indicator factor (how many confidence indicators we have)
        indicator_count = len(confidence_factors.get('indicators', []))
        indicator_factor = min(indicator_count / 5.0, 1.0)
        
        # Combine factors
        confidence = (base_confidence * 0.5 + competition_factor * 0.3 + indicator_factor * 0.2)
        
        return min(confidence, 1.0)
    
    def _generate_generic_prediction(self, content: str, headings: List[str], 
                                   scores: Dict[str, float]) -> Dict[str, Any]:
        """Generate generic prediction when no specific type is confident"""
        
        # Analyze content characteristics for generic persona selection
        content_lower = content.lower()
        
        # Count different types of terms
        analytical_terms = ['analysis', 'data', 'research', 'study', 'findings', 'insights']
        operational_terms = ['procedure', 'process', 'operation', 'management', 'coordination']
        strategic_terms = ['strategy', 'planning', 'vision', 'goals', 'objectives', 'direction']
        technical_terms = ['technical', 'system', 'technology', 'implementation', 'development']
        creative_terms = ['creative', 'design', 'art', 'content', 'story', 'brand']
        customer_terms = ['customer', 'client', 'service', 'support', 'relationship']
        
        term_counts = {
            'analytical': sum(1 for term in analytical_terms if term in content_lower),
            'operational': sum(1 for term in operational_terms if term in content_lower),
            'strategic': sum(1 for term in strategic_terms if term in content_lower),
            'technical': sum(1 for term in technical_terms if term in content_lower),
            'creative': sum(1 for term in creative_terms if term in content_lower),
            'customer_facing': sum(1 for term in customer_terms if term in content_lower)
        }
        
        if term_counts:
            dominant_type = max(term_counts, key=term_counts.get)
            if term_counts[dominant_type] > 0:
                return {
                    'document_type': 'generic_document',
                    'confidence': 0.5,
                    'persona': self.generic_personas[dominant_type],
                    'job_to_be_done': self.generic_jobs[dominant_type],
                    'all_scores': scores,
                    'confidence_factors': {
                        'generic_analysis': True,
                        'dominant_type': dominant_type,
                        'term_counts': term_counts
                    }
                }
        
        # Ultimate fallback
        return {
            'document_type': 'unknown',
            'confidence': 0.3,
            'persona': 'General professional (any role requiring document analysis)',
            'job_to_be_done': 'Extract relevant information and insights from documents',
            'all_scores': scores,
            'confidence_factors': {'fallback': True}
        }
    
    def predict_from_documents(self, document_sections: List[DocumentSection]) -> Dict[str, Any]:
        """Enhanced prediction from document sections with adaptive learning"""
        if not document_sections:
            return self._get_default_prediction()
        
        # Combine all document content
        all_content = ' '.join([section.text + ' ' + section.refined_text for section in document_sections])
        all_headings = [section.text for section in document_sections]
        
        # Get document type prediction
        prediction = self.predict_document_type(all_content, all_headings)
        
        # Enhance prediction with document-specific insights
        enhanced_prediction = self._enhance_prediction(prediction, document_sections)
        
        return enhanced_prediction
    
    def _enhance_prediction(self, base_prediction: Dict[str, Any], 
                          document_sections: List[DocumentSection]) -> Dict[str, Any]:
        """Enhanced prediction with adaptive insights and improved persona matching"""
        enhanced = base_prediction.copy()
        
        # Analyze document characteristics
        total_sections = len(document_sections)
        avg_section_length = np.mean([len(section.text) for section in document_sections]) if document_sections else 0
        
        # Enhanced term analysis
        all_text = ' '.join([section.text + ' ' + section.refined_text for section in document_sections])
        all_text_lower = all_text.lower()
        
        job_title_matches = self._match_job_title_patterns(all_text_lower) 
        industry_scores = self._detect_industry(all_text_lower)
        experience_level = self._analyze_experience_level(all_text_lower)
        
        # Multi-domain analysis (enhanced)
        domain_scores = {
            'technical': sum(1 for term in ['api', 'sdk', 'integration', 'code', 'function', 'parameter', 'endpoint', 'system', 'architecture', 'software', 'hardware', 'programming'] if term in all_text_lower),
            'business': sum(1 for term in ['business', 'strategy', 'market', 'customer', 'revenue', 'growth', 'partnership', 'stakeholder', 'executive', 'management'] if term in all_text_lower),
            'academic': sum(1 for term in ['research', 'study', 'methodology', 'analysis', 'findings', 'conclusion', 'hypothesis', 'academic', 'scientific'] if term in all_text_lower),
            'creative': sum(1 for term in ['design', 'creative', 'content', 'brand', 'story', 'visual', 'art', 'aesthetic', 'user experience'] if term in all_text_lower),
            'operational': sum(1 for term in ['procedure', 'process', 'operation', 'management', 'coordination', 'workflow', 'logistics', 'operations'] if term in all_text_lower)
        }
        
        clustering_analysis = self._ensemble_clustering_analysis(all_text, document_sections)
        contextual_scoring = self._contextual_scoring_system(all_text_lower, base_prediction)
        
        enhanced_persona = self._predict_enhanced_persona(
            base_prediction, job_title_matches, industry_scores, 
            experience_level, domain_scores, all_text_lower
        )
        
        if enhanced_persona:
            enhanced['persona'] = enhanced_persona['persona']
            enhanced['job_to_be_done'] = enhanced_persona['job_to_be_done']
            enhanced['confidence'] = min(enhanced['confidence'] + enhanced_persona['confidence_boost'], 1.0)
        
        # Apply additional confidence boost from contextual scoring
        if contextual_scoring and 'confidence_boost' in contextual_scoring:
            enhanced['confidence'] = min(enhanced['confidence'] + contextual_scoring['confidence_boost'], 1.0)
        
        # Add comprehensive confidence indicators
        enhanced['confidence_indicators'] = {
            'total_sections': total_sections,
            'avg_section_length': avg_section_length,
            'domain_scores': domain_scores,
            'job_title_matches': job_title_matches,
            'industry_scores': industry_scores,
            'experience_level': experience_level,
            'clustering_analysis': clustering_analysis,
            'contextual_scoring': contextual_scoring,
            'document_complexity': self._assess_complexity(all_text)
        }
        
        return enhanced
    
    def _match_job_title_patterns(self, text: str) -> Dict[str, Any]:
        """PHASE 1: Enhanced job title pattern matching"""
        matches = {}
        
        for persona_type, patterns in self.job_title_patterns.items():
            score = 0
            found_titles = []
            found_contexts = []
            found_responsibilities = []
            
            # Check for job titles
            for title in patterns['titles']:
                if title.lower() in text:
                    score += 5
                    found_titles.append(title)
            
            # Check for context words
            for context in patterns['contexts']:
                if context.lower() in text:
                    score += 2
                    found_contexts.append(context)
            
            # Check for responsibility keywords
            for responsibility in patterns['responsibilities']:
                if responsibility.lower() in text:
                    score += 3
                    found_responsibilities.append(responsibility)
            
            if score > 0:
                matches[persona_type] = {
                    'score': score,
                    'titles': found_titles,
                    'contexts': found_contexts,
                    'responsibilities': found_responsibilities
                }
        
        return matches
    
    def _detect_industry(self, text: str) -> Dict[str, float]:
        """PHASE 1: Industry-specific keyword detection"""
        industry_scores = {}
        
        for industry, keywords in self.industry_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword.lower() in text:
                    # Weight by frequency
                    frequency = text.count(keyword.lower())
                    score += frequency * 2
            
            if score > 0:
                industry_scores[industry] = score
        
        return industry_scores
    
    def _analyze_experience_level(self, text: str) -> Dict[str, Any]:
        """PHASE 1: Experience level detection"""
        experience_scores = {}
        
        for level, patterns in self.experience_patterns.items():
            score = 0
            found_indicators = []
            found_contexts = []
            
            # Check for experience indicators
            for indicator in patterns['indicators']:
                if indicator.lower() in text:
                    score += 4
                    found_indicators.append(indicator)
            
            # Check for context words
            for context in patterns['contexts']:
                if context.lower() in text:
                    score += 2
                    found_contexts.append(context)
            
            if score > 0:
                experience_scores[level] = {
                    'score': score,
                    'indicators': found_indicators,
                    'contexts': found_contexts
                }
        
        return experience_scores
    
    def _predict_enhanced_persona(self, base_prediction: Dict[str, Any], 
                                job_title_matches: Dict[str, Any],
                                industry_scores: Dict[str, float],
                                experience_level: Dict[str, Any],
                                domain_scores: Dict[str, int],
                                text: str) -> Optional[Dict[str, Any]]:
        """Enhanced persona prediction using multiple signals"""
        
        best_job_match = None
        if job_title_matches:
            best_job_match = max(job_title_matches.items(), key=lambda x: x[1]['score'])
        
        best_industry = None
        if industry_scores:
            best_industry = max(industry_scores.items(), key=lambda x: x[1])
        
        best_experience = None
        if experience_level:
            best_experience = max(experience_level.items(), key=lambda x: x[1]['score'])
        
        # Find the strongest domain
        best_domain = max(domain_scores.items(), key=lambda x: x[1]) if domain_scores else None
        
        # Generate enhanced persona based on strongest signals
        if best_job_match and best_job_match[1]['score'] > 8:
            persona_type = best_job_match[0]
            confidence_boost = 0.15
            
            # Customize persona based on industry and experience
            persona_base = {
                'executive': 'Executive professional',
                'technical': 'Technical professional', 
                'creative': 'Creative professional',
                'business': 'Business professional',
                'sales_marketing': 'Sales & Marketing professional',
                'research': 'Research professional'
            }.get(persona_type, 'Professional')
            
            # Add industry context if available
            if best_industry and best_industry[1] > 5:
                industry_name = best_industry[0].replace('_', ' ').title()
                persona_base = f"{persona_base} in {industry_name}"
            
            # Add experience level if available
            if best_experience and best_experience[1]['score'] > 6:
                exp_level = best_experience[0].replace('_', ' ').title()
                persona_base = f"{exp_level} {persona_base}"
            
            # Generate job-to-be-done based on persona type
            job_mappings = {
                'executive': 'Make strategic decisions and lead organizational initiatives',
                'technical': 'Implement, develop, and maintain technical solutions',
                'creative': 'Create compelling designs and innovative content',
                'business': 'Manage operations and drive business growth',
                'sales_marketing': 'Drive revenue growth and customer engagement',
                'research': 'Conduct research and generate actionable insights'
            }
            
            return {
                'persona': persona_base,
                'job_to_be_done': job_mappings.get(persona_type, 'Execute professional responsibilities effectively'),
                'confidence_boost': confidence_boost
            }
        
        return None
    
    def _ensemble_clustering_analysis(self, text: str, sections: List[DocumentSection]) -> Dict[str, Any]:
        """PHASE 2: Ensemble clustering for better persona grouping"""
        if len(sections) < 3:
            return {}
        
        try:
            texts = [section.text + ' ' + section.refined_text for section in sections]
            
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(texts)
            svd = TruncatedSVD(n_components=min(50, len(texts)-1))
            reduced_features = svd.fit_transform(tfidf_matrix)
            
            # Apply multiple clustering algorithms
            clustering_results = {}
            
            # K-Means clustering
            if len(texts) >= 3:
                kmeans = KMeans(n_clusters=min(3, len(texts)), random_state=42, n_init=10)
                kmeans_labels = kmeans.fit_predict(reduced_features)
                clustering_results['kmeans'] = kmeans_labels
            
            # DBSCAN clustering
            if len(texts) >= 5:
                dbscan = DBSCAN(eps=0.5, min_samples=2)
                dbscan_labels = dbscan.fit_predict(reduced_features)
                clustering_results['dbscan'] = dbscan_labels
            
            # Hierarchical clustering
            if len(texts) >= 3:
                hierarchical = AgglomerativeClustering(n_clusters=min(3, len(texts)))
                hierarchical_labels = hierarchical.fit_predict(reduced_features)
                clustering_results['hierarchical'] = hierarchical_labels
            
            # Analyze cluster characteristics
            cluster_analysis = self._analyze_clusters(texts, clustering_results, vectorizer)
            
            return {
                'cluster_results': clustering_results,
                'cluster_analysis': cluster_analysis,
                'feature_importance': self._get_feature_importance(vectorizer, reduced_features)
            }
            
        except Exception as e:
            logger.warning(f"Ensemble clustering failed: {e}")
            return {}
    
    def _analyze_clusters(self, texts: List[str], clustering_results: Dict[str, Any], 
                         vectorizer: TfidfVectorizer) -> Dict[str, Any]:
        """Analyze cluster characteristics for persona insights"""
        cluster_personas = {}
        
        for method, labels in clustering_results.items():
            unique_labels = set(labels)
            method_analysis = {}
            
            for label in unique_labels:
                if label == -1:  # Noise cluster in DBSCAN
                    continue
                    
                # Get texts in this cluster
                cluster_texts = [texts[i] for i, l in enumerate(labels) if l == label]
                combined_text = ' '.join(cluster_texts).lower()
                
                # Analyze cluster for persona indicators
                persona_scores = {}
                for persona_type, patterns in self.job_title_patterns.items():
                    score = 0
                    for title in patterns['titles']:
                        if title.lower() in combined_text:
                            score += 3
                    for context in patterns['contexts']:
                        if context.lower() in combined_text:
                            score += 1
                    persona_scores[persona_type] = score
                
                # Find dominant persona for this cluster
                if persona_scores:
                    dominant_persona = max(persona_scores, key=persona_scores.get)
                    method_analysis[f'cluster_{label}'] = {
                        'dominant_persona': dominant_persona,
                        'persona_scores': persona_scores,
                        'text_count': len(cluster_texts)
                    }
            
            cluster_personas[method] = method_analysis
        
        return cluster_personas
    
    def _get_feature_importance(self, vectorizer: TfidfVectorizer, features: np.ndarray) -> List[Tuple[str, float]]:
        """Get most important features from TF-IDF analysis"""
        try:
            feature_names = vectorizer.get_feature_names_out()
            mean_scores = np.mean(features, axis=0)
            
            # Get top features
            top_indices = np.argsort(mean_scores)[-20:][::-1]
            top_features = [(feature_names[i], float(mean_scores[i])) for i in top_indices]
            
            return top_features
        except Exception:
            return []
    
    def _contextual_scoring_system(self, text: str, base_prediction: Dict[str, Any]) -> Dict[str, Any]:
        """PHASE 2: Advanced contextual scoring for persona refinement"""
        
        # Context weights for different signals
        context_weights = {
            'job_title_context': 0.3,
            'responsibility_context': 0.25,
            'skill_context': 0.2,
            'industry_context': 0.15,
            'experience_context': 0.1
        }
        
        context_scores = {}
        text_lower = text.lower()
        
        # Job title context analysis
        job_title_score = 0
        for persona_type, patterns in self.job_title_patterns.items():
            for title in patterns['titles']:
                if title.lower() in text_lower:
                    # Weight by position in text (earlier = more important)
                    position = text_lower.find(title.lower())
                    position_weight = max(0.5, 1.0 - (position / len(text_lower)))
                    job_title_score += 5 * position_weight
        context_scores['job_title_context'] = job_title_score * context_weights['job_title_context']
        
        # Responsibility context analysis
        responsibility_score = 0
        responsibility_keywords = ['responsible for', 'manages', 'leads', 'oversees', 'develops', 'implements', 'coordinates']
        for keyword in responsibility_keywords:
            if keyword in text_lower:
                responsibility_score += 3
        context_scores['responsibility_context'] = responsibility_score * context_weights['responsibility_context']
        
        # Skill context analysis
        skill_score = 0
        skill_keywords = ['experience with', 'skilled in', 'expertise in', 'proficient in', 'knowledge of', 'familiar with']
        for keyword in skill_keywords:
            if keyword in text_lower:
                skill_score += 2
        context_scores['skill_context'] = skill_score * context_weights['skill_context']
        
        # Industry context analysis
        industry_score = 0
        for industry, keywords in self.industry_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    industry_score += 1
        context_scores['industry_context'] = industry_score * context_weights['industry_context']
        
        # Experience context analysis
        experience_score = 0
        for level, patterns in self.experience_patterns.items():
            for indicator in patterns['indicators']:
                if indicator.lower() in text_lower:
                    experience_score += 2
        context_scores['experience_context'] = experience_score * context_weights['experience_context']
        
        # Calculate overall contextual confidence
        total_contextual_score = sum(context_scores.values())
        contextual_confidence = min(total_contextual_score / 10.0, 1.0)  # Normalize to 0-1
        
        return {
            'context_scores': context_scores,
            'contextual_confidence': contextual_confidence,
            'confidence_boost': contextual_confidence * 0.2  # Up to 20% boost
        }
    
    def _assess_complexity(self, text: str) -> str:
        """Assess document complexity level"""
        word_count = len(text.split())
        technical_terms = len([word for word in text.lower().split() 
                             if word in ['api', 'sdk', 'integration', 'protocol', 'algorithm']])
        
        if technical_terms > 10 or word_count > 5000:
            return 'high'
        elif technical_terms > 5 or word_count > 2000:
            return 'medium'
        else:
            return 'low'
    
    def _get_default_prediction(self) -> Dict[str, Any]:
        """Get default prediction when no documents are available"""
        return {
            'document_type': 'unknown',
            'confidence': 0.0,
            'persona': 'General professional (any role requiring document analysis)',
            'job_to_be_done': 'Extract relevant information and insights from documents',
            'all_scores': {},
            'confidence_factors': {}
        }

# ==================== Core Data Models ====================

@dataclass
class RankedSection:
    """Represents a ranked section with importance score"""
    section: DocumentSection
    importance_score: float
    rank: int
    relevance_factors: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'document': self.section.document_name,
            'page': self.section.page,
            'section_title': self.section.text,
            'level': f"H{getattr(self.section.heading_info, 'level', 1)}",
            'importance_rank': self.rank,
            'importance_score': self.importance_score,
            'relevance_factors': self.relevance_factors
        }

@dataclass
class AnalysisQuery:
    """Represents an analysis query with persona and job context"""
    persona: str
    job_to_be_done: str
    additional_context: Dict[str, Any] = field(default_factory=dict)
    
    def get_query_text(self) -> str:
        """Get combined query text"""
        return f"{self.persona} {self.job_to_be_done}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'persona': self.persona,
            'job_to_be_done': self.job_to_be_done,
            'additional_context': self.additional_context
        }

@dataclass
class AnalysisResult:
    """Complete analysis result with metadata"""
    query: AnalysisQuery
    ranked_sections: List[RankedSection]
    document_summaries: List[Dict[str, Any]]
    processing_metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'metadata': {
                'input_documents': [doc['name'] for doc in self.document_summaries],
                'persona': self.query.persona,
                'job_to_be_done': self.query.job_to_be_done,
                'processing_timestamp': datetime.utcnow().isoformat() + 'Z',
                **self.processing_metadata
            },
            'extracted_sections': [section.to_dict() for section in self.ranked_sections],
            'subsection_analysis': self._build_subsection_analysis(),
            'document_summaries': self.document_summaries
        }
    
    def _build_subsection_analysis(self):
        # Improved refined_text: extract text between headings using pdf_loader
        results = []
        doc_to_spans = {}
        for section in self.ranked_sections:
            doc = section.section.document_name
            if doc not in doc_to_spans:
                pdf_path = None
                for summary in self.document_summaries:
                    if summary['name'] == doc:
                        pdf_path = summary['path']
                        break
                if pdf_path:
                    try:
                        doc_to_spans[doc] = pdf_loader.extract_text_spans(pdf_path)
                    except Exception:
                        doc_to_spans[doc] = []
                else:
                    doc_to_spans[doc] = []
        for idx, section in enumerate(self.ranked_sections):
            doc = section.section.document_name
            page = section.section.page
            heading_text = section.section.text
            level = getattr(section.section.heading_info, 'level', 1)
            # Find all headings in this doc and page
            headings_on_page = [s for s in self.ranked_sections if s.section.document_name == doc and s.section.page == page]
            # Find this heading's position among headings on this page
            this_idx = [i for i, s in enumerate(headings_on_page) if s.section.text == heading_text][0]
            # Get all spans for this doc and page
            spans = [s for s in doc_to_spans.get(doc, []) if s.page == page+1]  # spans are 1-based
            # Find the index of the heading span
            heading_span_idx = next((i for i, s in enumerate(spans) if s.text.strip() == heading_text.strip()), None)
            # Find the next heading span index (if any)
            next_heading_span_idx = None
            if this_idx+1 < len(headings_on_page):
                next_heading_text = headings_on_page[this_idx+1].section.text
                next_heading_span_idx = next((i for i, s in enumerate(spans) if s.text.strip() == next_heading_text.strip()), None)
            # Extract text between heading_span_idx and next_heading_span_idx
            if heading_span_idx is not None:
                if next_heading_span_idx is not None and next_heading_span_idx > heading_span_idx:
                    content_spans = spans[heading_span_idx+1:next_heading_span_idx]
                else:
                    content_spans = spans[heading_span_idx+1:]
                refined_text = ' '.join(s.text for s in content_spans).strip()
            else:
                refined_text = section.section.text  # fallback
            results.append({
                'document': doc,
                'refined_text': refined_text if refined_text else section.section.text,
                'page': page,
                'level': f"H{level}",
                'importance_score': section.importance_score
            })
        return results

# ==================== Strategy Pattern for Ranking ====================

class RankingStrategy(ABC):
    """Abstract base class for section ranking strategies"""
    
    @abstractmethod
    def rank_sections(self, sections: List[DocumentSection], 
                     query: AnalysisQuery) -> List[RankedSection]:
        """Rank sections based on relevance to query"""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get strategy identifier"""
        pass

class TfidfRankingStrategy(RankingStrategy):
    """TF-IDF based ranking strategy with query expansion and cosine similarity"""
    
    def __init__(self, max_features: int = 10000, min_df: int = 1, max_df: float = 0.95):
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
    
    def _expand_query(self, query: AnalysisQuery) -> str:
        """Expand the query with persona/job keywords and document type patterns for better TF-IDF overlap."""
        base_query = query.get_query_text()
        # Try to get document type patterns from AdaptiveDocumentTypePredictor
        try:
            predictor = AdaptiveDocumentTypePredictor()
            # Use persona/job to guess document type
            doc_type = None
            for k, v in predictor.document_patterns.items():
                if query.persona and v['persona'].lower() in query.persona.lower():
                    doc_type = k
                    break
            if not doc_type:
                for k, v in predictor.document_patterns.items():
                    if query.job_to_be_done and v['job_to_be_done'].lower() in query.job_to_be_done.lower():
                        doc_type = k
                        break
            expansion = []
            if doc_type:
                expansion += predictor.document_patterns[doc_type]['keywords']
                expansion += predictor.document_patterns[doc_type]['patterns']
                expansion += predictor.document_patterns[doc_type]['semantic_themes']
            return base_query + ' ' + ' '.join(expansion)
        except Exception as e:
            logger.warning(f"[TFIDF] Query expansion failed: {e}")
            return base_query
    
    def rank_sections(self, sections: List[DocumentSection], 
                     query: AnalysisQuery) -> List[RankedSection]:
        """Rank sections using expanded TF-IDF query and cosine similarity"""
        if not sections:
            return []
        try:
            # Use both text and refined_text for each section
            section_texts = [
                (section.text + ' ' + section.refined_text).strip() if section.refined_text else section.text.strip()
                for section in sections
            ]
            # Expand the query
            expanded_query = self._expand_query(query)
            logger.info(f"[TFIDF] Expanded query: {expanded_query}")
            # Fit TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                min_df=self.min_df,
                max_df=self.max_df,
                stop_words='english'
            )
            all_texts = section_texts + [expanded_query]
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            section_vectors = tfidf_matrix[:-1]
            query_vector = tfidf_matrix[-1]
            # Use cosine similarity
            similarity_scores = cosine_similarity(section_vectors, query_vector).flatten()
            # Create ranked sections
            ranked_sections = []
            for i, (section, score) in enumerate(zip(sections, similarity_scores)):
                ranked_section = RankedSection(
                    section=section,
                    importance_score=float(score),
                    rank=i + 1,
                    relevance_factors={'tfidf_similarity': float(score)}
                )
                ranked_sections.append(ranked_section)
            # Sort by importance score
            ranked_sections.sort(key=lambda x: x.importance_score, reverse=True)
            # Update ranks
            for i, section in enumerate(ranked_sections):
                section.rank = i + 1
            logger.info(f"[TFIDF] Top 3 tfidf_similarity scores: {[s.importance_score for s in ranked_sections[:3]]}")
            return ranked_sections
        except Exception as e:
            logger.error(f"Error in TF-IDF ranking: {e}")
            return []
    
    def get_strategy_name(self) -> str:
        return "tfidf_ranking"

class BM25RankingStrategy(RankingStrategy):
    """BM25 based ranking strategy with query expansion"""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.tokenizer = nltk.word_tokenize
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into terms"""
        tokens = self.tokenizer(text.lower())
        return [t for t in tokens if t.isalnum()]
    
    def _expand_query(self, query: AnalysisQuery) -> str:
        """Reuse query expansion from TF-IDF strategy"""
        tfidf_strategy = TfidfRankingStrategy()
        return tfidf_strategy._expand_query(query)
    
    def rank_sections(self, sections: List[DocumentSection], 
                     query: AnalysisQuery) -> List[RankedSection]:
        """Rank sections using BM25 algorithm with query expansion"""
        if not sections:
            return []
            
        try:
            # Prepare document texts
            section_texts = [
                (section.text + ' ' + section.refined_text).strip() 
                if section.refined_text else section.text.strip()
                for section in sections
            ]
            
            # Tokenize documents
            tokenized_docs = [self._tokenize(doc) for doc in section_texts]
            
            # Create BM25 index
            bm25 = BM25Okapi(tokenized_docs, k1=self.k1, b=self.b)
            
            # Expand and tokenize query
            expanded_query = self._expand_query(query)
            logger.info(f"[BM25] Expanded query: {expanded_query}")
            query_tokens = self._tokenize(expanded_query)
            
            # Get scores
            doc_scores = bm25.get_scores(query_tokens)
            
            # Create ranked sections
            ranked_sections = []
            for i, (section, score) in enumerate(zip(sections, doc_scores)):
                ranked_section = RankedSection(
                    section=section,
                    importance_score=float(score),
                    rank=i + 1,
                    relevance_factors={'bm25_score': float(score)}
                )
                ranked_sections.append(ranked_section)
            
            # Sort by BM25 score (descending)
            ranked_sections.sort(key=lambda x: x.importance_score, reverse=True)
            
            # Update ranks
            for i, section in enumerate(ranked_sections):
                section.rank = i + 1
                
            # Log top scores for debugging
            if ranked_sections:
                top_scores = [f"{s.importance_score:.4f}" for s in ranked_sections[:5]]
                logger.info(f"[BM25] Top 5 BM25 scores: {', '.join(top_scores)}")
            
            return ranked_sections
            
        except Exception as e:
            logger.error(f"Error in BM25 ranking: {e}")
            logger.debug(f"Error details:", exc_info=True)
            return []
    
    def get_strategy_name(self) -> str:
        return "bm25_ranking"

class EnhancedRankingStrategy(RankingStrategy):
    """Enhanced ranking strategy combining multiple factors"""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            'tfidf_similarity': 0.4,
            'section_length': 0.1,
            'page_position': 0.1,
            'heading_level': 0.2,
            'font_size': 0.2
        }
        self.tfidf_strategy = TfidfRankingStrategy()
    
    def rank_sections(self, sections: List[DocumentSection], 
                     query: AnalysisQuery) -> List[RankedSection]:
        """Rank sections using multiple factors"""
        if not sections:
            return []
        
        # Get TF-IDF scores
        tfidf_ranked = self.tfidf_strategy.rank_sections(sections, query)
        tfidf_scores = {section.section.text: section.importance_score 
                       for section in tfidf_ranked}
        
        # Calculate additional factors
        enhanced_sections = []
        for section in sections:
            factors = self._calculate_relevance_factors(section, tfidf_scores)
            
            # Combine scores
            combined_score = sum(
                factors.get(factor, 0) * weight 
                for factor, weight in self.weights.items()
            )
            
            enhanced_section = RankedSection(
                section=section,
                importance_score=combined_score,
                rank=0,  # Will be updated after sorting
                relevance_factors=factors
            )
            enhanced_sections.append(enhanced_section)
        
        # Sort and update ranks
        enhanced_sections.sort(key=lambda x: x.importance_score, reverse=True)
        for i, section in enumerate(enhanced_sections):
            section.rank = i + 1
        
        return enhanced_sections
    
    def _calculate_relevance_factors(self, section: DocumentSection, 
                                   tfidf_scores: Dict[str, float]) -> Dict[str, float]:
        """Calculate multiple relevance factors for a section"""
        factors = {}
        
        # TF-IDF similarity
        factors['tfidf_similarity'] = tfidf_scores.get(section.text, 0.0)
        
        # Section length factor (normalize text length)
        text_length = len(section.text)
        factors['section_length'] = min(text_length / 200, 1.0)  # Normalize to 0-1
        
        # Page position factor (earlier pages might be more important)
        factors['page_position'] = max(0.1, 1.0 - (section.page - 1) / 100)
        
        # Heading level factor (from heading info if available)
        if hasattr(section, 'heading_info') and section.heading_info:
            level = getattr(section.heading_info, 'level', 1)
            factors['heading_level'] = max(0.1, 1.0 - (level - 1) / 6)
        else:
            factors['heading_level'] = 0.5
        
        # Font size factor (if available in metadata)
        font_size = section.metadata.get('font_size', 12)
        factors['font_size'] = min(font_size / 18, 1.0)  # Normalize to 0-1
        
        return factors
    
    def get_strategy_name(self) -> str:
        return "enhanced_ranking"

# ==================== Document Processing Pipeline ====================

class DocumentProcessor:
    """Process documents to extract sections"""
    
    def __init__(self, outline_builder=None):
        self.outline_builder = outline_builder or OutlineBuilderFactory.create_enhanced_builder()
    
    def process_documents(self, pdf_paths: List[str]) -> Tuple[List[DocumentSection], List[Dict[str, Any]]]:
        """Process multiple documents and extract sections"""
        all_sections = []
        document_summaries = []
        
        for pdf_path in pdf_paths:
            try:
                sections, summary = self._process_single_document(pdf_path)
                all_sections.extend(sections)
                document_summaries.append(summary)
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {e}")
                continue
        
        return all_sections, document_summaries
    
    def _process_single_document(self, pdf_path: str) -> Tuple[List[DocumentSection], Dict[str, Any]]:
        """Process a single document"""
        outline = self.outline_builder.build_outline(pdf_path)
        document_name = os.path.basename(pdf_path)
        # Extract all spans for this document
        try:
            all_spans = pdf_loader.extract_text_spans(pdf_path)
        except Exception:
            all_spans = []
        sections = []
        headings = outline.headings
        for i, heading in enumerate(headings):
            page = heading.page
            heading_text = heading.text
            # Get all spans for this page
            spans = [s for s in all_spans if s.page == page+1]  # spans are 1-based
            # Find the index of the heading span
            heading_span_idx = next((j for j, s in enumerate(spans) if s.text.strip() == heading_text.strip()), None)
            # Find the next heading span index (if any)
            next_heading_span_idx = None
            if i+1 < len(headings) and headings[i+1].page == page:
                next_heading_text = headings[i+1].text
                next_heading_span_idx = next((j for j, s in enumerate(spans) if s.text.strip() == next_heading_text.strip()), None)
            # Extract text between heading_span_idx and next_heading_span_idx
            if heading_span_idx is not None:
                if next_heading_span_idx is not None and next_heading_span_idx > heading_span_idx:
                    content_spans = spans[heading_span_idx+1:next_heading_span_idx]
                else:
                    content_spans = spans[heading_span_idx+1:]
                refined_text = ' '.join(s.text for s in content_spans).strip()
            else:
                refined_text = heading_text  # fallback
            section = DocumentSection(
                text=heading_text,
                document_name=document_name,
                page=page,
                heading_info=heading,
                confidence=heading.confidence,
                metadata=heading.metadata,
                refined_text=refined_text if refined_text else heading_text
            )
            sections.append(section)
        summary = {
            'name': document_name,
            'path': pdf_path,
            'title': outline.title,
            'total_headings': len(outline.headings),
            'processing_time': outline.processing_time,
            'confidence_score': outline.confidence_score,
            'document_type': outline.document_type
        }
        return sections, summary

# ==================== Main Document Analyzer ====================

class DocumentAnalyzerConfig:
    """Configuration for document analyzer"""
    
    def __init__(self):
        self.ranking_strategy: RankingStrategy = TfidfRankingStrategy()
        self.max_results: int = 5  # Default to top 5 results
        self.min_confidence: float = 0.1
        self.enable_caching: bool = True
        self.outline_builder = None

class DocumentAnalyzer:
    """Main document analyzer with configurable strategies"""
    
    def __init__(self, config: Optional[DocumentAnalyzerConfig] = None):
        self.config = config or DocumentAnalyzerConfig()
        self.document_processor = DocumentProcessor(self.config.outline_builder)
        self._cache: Dict[str, AnalysisResult] = {}
        self.predictor = AdaptiveDocumentTypePredictor()  # Use enhanced predictor
    
    def analyze_documents(self, pdf_paths: List[str], persona: str = None, 
                         job: str = None, **kwargs) -> AnalysisResult:
        """Analyze documents for relevance to persona and job"""
        start_time = datetime.utcnow()
        
        # Process documents first to get sections for prediction
        sections, document_summaries = self.document_processor.process_documents(pdf_paths)
        
        # Auto-predict persona and job if not provided
        if persona is None or job is None:
            prediction = self.predictor.predict_from_documents(sections)
            if persona is None:
                persona = prediction['persona']
                logger.info(f"Auto-predicted persona: {persona} (confidence: {prediction['confidence']:.2f})")
            if job is None:
                job = prediction['job_to_be_done']
                logger.info(f"Auto-predicted job-to-be-done: {job} (confidence: {prediction['confidence']:.2f})")
            
            # Add prediction metadata to kwargs
            kwargs['auto_prediction'] = prediction
        
        # Create analysis query
        query = AnalysisQuery(
            persona=persona,
            job_to_be_done=job,
            additional_context=kwargs
        )
        
        # Check cache
        cache_key = self._generate_cache_key(pdf_paths, query)
        if self.config.enable_caching and cache_key in self._cache:
            logger.info(f"Returning cached result for query: {persona} / {job}")
            return self._cache[cache_key]
        
        try:
            
            # Filter sections by confidence
            filtered_sections = [
                section for section in sections 
                if section.confidence >= self.config.min_confidence
            ]
            
            # Rank sections
            ranked_sections = self.config.ranking_strategy.rank_sections(
                filtered_sections, query
            )
            
            # Take top max_results sections overall
            top_sections = ranked_sections[:self.config.max_results]
            
            # Create result
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            result = AnalysisResult(
                query=query,
                ranked_sections=top_sections,
                document_summaries=document_summaries,
                processing_metadata={
                    'total_sections_found': len(sections),
                    'sections_after_filtering': len(filtered_sections),
                    'ranking_strategy': self.config.ranking_strategy.get_strategy_name(),
                    'processing_time_seconds': processing_time,
                    'max_results': self.config.max_results
                }
            )
            
            # Cache result
            if self.config.enable_caching:
                self._cache[cache_key] = result
            
            logger.info(f"Analysis completed: {len(top_sections)} sections ranked "
                       f"in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing documents: {e}")
            raise DocumentAnalysisException(f"Analysis failed: {e}") from e
    
    def analyze_documents_auto(self, pdf_paths: List[str], **kwargs) -> AnalysisResult:
        """Analyze documents with automatic persona and job-to-be-done prediction"""
        return self.analyze_documents(pdf_paths, persona=None, job=None, **kwargs)
    
    def analyze_documents_batch(self, requests: List[Dict[str, Any]]) -> List[AnalysisResult]:
        """Analyze multiple document sets in batch"""
        results = []
        
        for request in requests:
            try:
                result = self.analyze_documents(
                    pdf_paths=request['pdf_paths'],
                    persona=request['persona'],
                    job=request['job'],
                    **request.get('kwargs', {})
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error in batch analysis: {e}")
                continue
        
        return results
    
    def _generate_cache_key(self, pdf_paths: List[str], query: AnalysisQuery) -> str:
        """Generate cache key for analysis request"""
        import hashlib
        
        # Create key from paths and query
        key_data = {
            'pdf_paths': sorted(pdf_paths),
            'persona': query.persona,
            'job': query.job_to_be_done,
            'strategy': self.config.ranking_strategy.get_strategy_name()
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

# ==================== Factory for Easy Configuration ====================

class DocumentAnalyzerFactory:
    """Factory for creating pre-configured document analyzers"""
    
    @staticmethod
    def create_basic_analyzer() -> DocumentAnalyzer:
        """Create basic document analyzer"""
        config = DocumentAnalyzerConfig()
        return DocumentAnalyzer(config)
    
    @staticmethod
    def create_enhanced_analyzer() -> DocumentAnalyzer:
        """Create enhanced analyzer with multi-factor ranking"""
        config = DocumentAnalyzerConfig()
        config.ranking_strategy = EnhancedRankingStrategy()
        config.max_results = 15
        config.outline_builder = OutlineBuilderFactory.create_enhanced_builder()
        return DocumentAnalyzer(config)
    
    @staticmethod
    def create_high_performance_analyzer() -> DocumentAnalyzer:
        """Create high-performance analyzer with caching"""
        config = DocumentAnalyzerConfig()
        config.ranking_strategy = EnhancedRankingStrategy()
        config.enable_caching = True
        config.max_results = 20
        config.outline_builder = OutlineBuilderFactory.create_high_performance_builder()
        return DocumentAnalyzer(config)

# ==================== Custom Exceptions ====================

class DocumentAnalysisException(Exception):
    """Base exception for document analysis"""
    pass

# ==================== Convenience Functions ====================

def analyze_documents(pdf_paths: List[str], persona: str = None, job: str = None, 
                     **kwargs) -> Dict[str, Any]:
    """
    Convenience function for document analysis with optional auto-prediction.
    
    Args:
        pdf_paths: List of PDF file paths
        persona: User persona description (auto-predicted if None)
        job: Job to be done description (auto-predicted if None)
        **kwargs: Additional configuration options
        
    Returns:
        Dictionary representation of analysis results
    """
    # Create analyzer based on kwargs
    if kwargs.get('enhanced', False):
        analyzer = DocumentAnalyzerFactory.create_enhanced_analyzer()
    else:
        analyzer = DocumentAnalyzerFactory.create_basic_analyzer()
    
    # Perform analysis
    result = analyzer.analyze_documents(pdf_paths, persona, job, **kwargs)
    
    # Return dictionary representation
    return result.to_dict()

def analyze_documents_enhanced(pdf_paths: List[str], persona: str, job: str,
                              ranking_weights: Optional[Dict[str, float]] = None,
                              max_results: int = 10) -> Dict[str, Any]:
    """
    Enhanced analysis function with custom configuration.
    
    Args:
        pdf_paths: List of PDF file paths
        persona: User persona description
        job: Job to be done description
        ranking_weights: Custom weights for ranking factors
        max_results: Maximum number of results to return
        
    Returns:
        Dictionary representation of analysis results
    """
    config = DocumentAnalyzerConfig()
    config.ranking_strategy = EnhancedRankingStrategy(weights=ranking_weights)
    config.max_results = max_results
    config.outline_builder = OutlineBuilderFactory.create_enhanced_builder()
    
    analyzer = DocumentAnalyzer(config)
    result = analyzer.analyze_documents(pdf_paths, persona, job)
    
    return result.to_dict()

def analyze_documents_auto(pdf_paths: List[str], **kwargs) -> Dict[str, Any]:
    """
    Convenience function for automatic document analysis.
    
    Args:
        pdf_paths: List of PDF file paths
        **kwargs: Additional configuration options
        
    Returns:
        Dictionary with analysis results including auto-predicted persona and job
    """
    analyzer = DocumentAnalyzerFactory.create_enhanced_analyzer()
    return analyzer.analyze_documents_auto(pdf_paths, **kwargs).to_dict()

# ==================== Usage Examples ====================

def demo_usage():
    """Demonstrate various usage patterns"""
    
    print("=== Enhanced Document Analyzer Demo ===\n")
    
    # Sample data
    pdf_paths = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
    persona = "Software Engineer"
    job = "API Integration"
    
    # 1. Basic usage (backward compatible)
    print("1. Basic Usage:")
    
    print("\n2. Enhanced Usage:")
    custom_weights = {
        'tfidf_similarity': 0.5,
        'section_length': 0.2,
        'page_position': 0.1,
        'heading_level': 0.1,
        'font_size': 0.1
    }
    
    # 3. Object-oriented usage
    print("\n3. Object-Oriented Usage:")
    analyzer = DocumentAnalyzerFactory.create_enhanced_analyzer()
    # result = analyzer.analyze_documents(pdf_paths, persona, job)
    
    # 4. Batch processing
    print("\n4. Batch Processing:")
    requests = [
        {'pdf_paths': pdf_paths, 'persona': 'Data Scientist', 'job': 'Model Training'},
        {'pdf_paths': pdf_paths, 'persona': 'Product Manager', 'job': 'Feature Planning'}
    ]
    
    print("\n5. Automatic Prediction:")

    
    print("\nDemo completed!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Persona-driven PDF document analyzer")
    parser.add_argument('--input', type=str, required=True, help='Input folder containing PDFs')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file path (required format)')
    parser.add_argument('--debug_output', type=str, default=None, help='Debug output JSON file path (detailed output)')
    parser.add_argument('--enhanced', action='store_true', help='Use enhanced ranking strategy')
    # Remove persona, job, and auto arguments
    args = parser.parse_args()

    if args.debug_output is None:
        output_dir = os.path.dirname(args.output)
        debug_dir = os.path.join(output_dir, 'debug_output')
        os.makedirs(debug_dir, exist_ok=True)
        debug_output_path = os.path.join(debug_dir, 'detailed_output.json')
    else:
        debug_output_path = args.debug_output
        os.makedirs(os.path.dirname(debug_output_path), exist_ok=True)

    # Setting up logging to file in the same folder as debug_output
    log_dir = os.path.dirname(debug_output_path)
    log_file = os.path.join(log_dir, 'pipeline.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s:%(name)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("Logging is configured and pipeline.log should capture this message.")

    # Find all PDFs in the input folder
    pdf_paths = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.lower().endswith('.pdf')]
    pdf_paths.sort()

    # Read and concatenate all document texts for persona/job prediction
    document_text = ""
    for pdf_path in pdf_paths:
        try:
            with open(pdf_path, 'rb') as f:
                import fitz
                doc = fitz.open(stream=f.read(), filetype="pdf")
                for page in doc:
                    text = page.get_text()
                    if not text.strip():
                        # Fallback to OCR
                        pix = page.get_pixmap()
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        text = pytesseract.image_to_string(img)
                    document_text += text
            logger.info(f"Extracted text from {pdf_path}")
        except Exception as e:
            logger.warning(f"Failed to extract text from {pdf_path}: {e}")
            continue

    # Always use clustering/TF-IDF for persona/job extraction
    logger.info("[Dynamic] Using clustering to extract personas and jobs...")
    dynamic_personas, dynamic_jobs = extract_dynamic_personas_jobs(document_text, n_clusters=3)
    # Fallback if extraction fails
    if not dynamic_personas or not dynamic_jobs:
        logger.warning("[Warning] No personas or jobs could be extracted. Check if the input PDFs contain text.")
        persona, job = "Unknown Persona", "Unknown Job"
    else:
        persona, job = dynamic_personas[0], dynamic_jobs[0]
    persona_score, job_score = 1.0, 1.0  # Not meaningful for clustering, set to 1.0

    logger.info(f"Predicted Persona: {persona}")
    logger.info(f"Predicted Job: {job}")
    # Pass the predicted persona and job into the core analysis pipeline
    result_dict = analyze_documents(pdf_paths, persona, job, enhanced=args.enhanced)
    # Optionally, add all dynamic personas/jobs to debug output
    result_dict['dynamic_personas'] = dynamic_personas
    result_dict['dynamic_jobs'] = dynamic_jobs

    # Write detailed output to debug_output
    with open(debug_output_path, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)

    # Map to required format and write to main output
    import subprocess
    subprocess.run([
        'python',
        os.path.join(os.path.dirname(__file__), 'map_to_example_format.py'),
        debug_output_path,
        args.output
    ], check=True)

    # Ensure all logs are flushed to pipeline.log
    import logging
    logging.shutdown()
