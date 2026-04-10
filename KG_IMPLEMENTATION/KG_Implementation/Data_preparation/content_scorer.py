"""
Content-based paragraph scoring.
Scores each paragraph based on climate relevance and quality.
"""

from typing import Dict

from config import (
    ALL_CLIMATE_KEYWORDS, CAUSAL_PHRASES, IMPACT_PHRASES, FACTUAL_PHRASES,
    META_ORGANIZATIONAL_PHRASES, ACKNOWLEDGMENT_PHRASES, METHODOLOGY_PHRASES,
    AUTHOR_AFFILIATION_PHRASES, FIGURE_TABLE_PHRASES, SCORING_WEIGHTS,
    MIN_PARAGRAPH_LENGTH, MAX_PARAGRAPH_LENGTH,
    TOC_PATTERNS, LIST_INDICATORS, PAGE_REF_PATTERN  # NEW
)
from utils import count_words, contains_any_phrase, count_citations, has_quantitative_data


class ContentScorer:

    def __init__(self):
        self.weights = SCORING_WEIGHTS
    
    def calculate_keyword_density(self, text: str) -> float:

        text_lower = text.lower()
        words = text_lower.split()
        
        if not words:
            return 0.0
        
        keyword_count = sum(1 for word in words if word in ALL_CLIMATE_KEYWORDS)
        
        for keyword in ALL_CLIMATE_KEYWORDS:
            if ' ' in keyword and keyword in text_lower:
                # Count occurrences
                keyword_count += text_lower.count(keyword)
        
        density = keyword_count / len(words)
        return density
    
    def score_paragraph(self, paragraph: str) -> Dict:
        score = 50  # Base score (neutral)
        breakdown = {'base': 50}
        
        text_lower = paragraph.lower()
        word_count = count_words(paragraph)
        
        # ====================================================================
        # POSITIVE FACTORS
        # ====================================================================
        
        # 1. Climate keyword density
        keyword_density = self.calculate_keyword_density(paragraph)
        if keyword_density >= 0.15:
            points = self.weights['climate_keyword_density_high']
            score += points
            breakdown['climate_keywords'] = points
        elif keyword_density >= 0.10:
            points = self.weights['climate_keyword_density_medium']
            score += points
            breakdown['climate_keywords'] = points
        elif keyword_density >= 0.05:
            points = self.weights['climate_keyword_density_low']
            score += points
            breakdown['climate_keywords'] = points
        
        # 2. Causal language
        if contains_any_phrase(paragraph, CAUSAL_PHRASES):
            points = self.weights['causal_language']
            score += points
            breakdown['causal_language'] = points
        
        # 3. Quantitative data
        if has_quantitative_data(paragraph):
            points = self.weights['quantitative_data']
            score += points
            breakdown['quantitative_data'] = points
        
        # 4. Impact language
        if contains_any_phrase(paragraph, IMPACT_PHRASES):
            points = self.weights['impact_language']
            score += points
            breakdown['impact_language'] = points
        
        # 5. Factual tone
        if contains_any_phrase(paragraph, FACTUAL_PHRASES):
            points = self.weights['factual_tone']
            score += points
            breakdown['factual_tone'] = points
        
        # 6. Proper length
        if MIN_PARAGRAPH_LENGTH <= word_count <= MAX_PARAGRAPH_LENGTH:
            points = self.weights['proper_length']
            score += points
            breakdown['proper_length'] = points
        
        # ====================================================================
        # NEGATIVE FACTORS
        # ====================================================================
        
        # 1. Meta-organizational text
        if contains_any_phrase(paragraph, META_ORGANIZATIONAL_PHRASES):
            points = self.weights['meta_organizational']
            score += points  # points is negative
            breakdown['meta_organizational'] = points
        
        # 2. Heavy citations
        citation_count = count_citations(paragraph)
        if citation_count >= 3:
            points = self.weights['heavy_citations']
            score += points
            breakdown['heavy_citations'] = points
        
        # 3. Acknowledgment language
        if contains_any_phrase(paragraph, ACKNOWLEDGMENT_PHRASES):
            points = self.weights['acknowledgment']
            score += points
            breakdown['acknowledgment'] = points
        
        # 4. Methodology-heavy
        methodology_count = sum(1 for phrase in METHODOLOGY_PHRASES 
                               if phrase in text_lower)
        if methodology_count >= 2:
            points = self.weights['methodology_heavy']
            score += points
            breakdown['methodology_heavy'] = points
        
        # 5. Author affiliation
        if contains_any_phrase(paragraph, AUTHOR_AFFILIATION_PHRASES):
            points = self.weights['author_affiliation']
            score += points
            breakdown['author_affiliation'] = points
        
        # 6. Figure/table reference only
        if contains_any_phrase(paragraph, FIGURE_TABLE_PHRASES) and word_count < 20:
            points = self.weights['figure_table_only']
            score += points
            breakdown['figure_table_only'] = points
        
        # 7. Wrong length
        if word_count < MIN_PARAGRAPH_LENGTH or word_count > MAX_PARAGRAPH_LENGTH:
            points = self.weights['wrong_length']
            score += points
            breakdown['wrong_length'] = points
        
        # ====================================================================
        # FINAL SCORE
        # ====================================================================
        
        # Cap score between 0 and 100
        score = max(0, min(100, score))
        
        return {
            'score': score,
            'breakdown': breakdown,
            'word_count': word_count,
            'keyword_density': keyword_density
        }

def is_toc_or_list(self, text: str) -> bool:

    text_lower = text.lower()
    
    # Check for list indicators
    for indicator in LIST_INDICATORS:
        if indicator in text_lower:
            return True
    
    # Check for TOC patterns
    for pattern in TOC_PATTERNS:
        if re.search(pattern, text):
            return True
    
    # Check if line ends with page number reference
    lines = text.split('\n')
    page_ref_count = sum(1 for line in lines if re.search(PAGE_REF_PATTERN, line))
    
    # If more than 50% of lines have page references, it's a TOC
    if lines and (page_ref_count / len(lines)) > 0.5:
        return True
    
    return False

def calculate_dot_density(self, text: str) -> float:

    if not text:
        return 0.0
    
    dot_count = text.count('.')
    return dot_count / len(text)

def has_box_figure_table_refs(self, text: str) -> bool:

    patterns = [
        r'Box\s+\d+\.\d+',
        r'Figure\s+\d+\.\d+',
        r'Table\s+\d+\.\d+',
        r'Fig\.\s+\d+\.\d+',
    ]
    
    ref_count = 0
    for pattern in patterns:
        ref_count += len(re.findall(pattern, text, re.IGNORECASE))
    
    return ref_count > 2

# Example usage
if __name__ == '__main__':
    scorer = ContentScorer()
    
    # Good paragraph
    good_para = """
    Global warming is primarily caused by greenhouse gas emissions from burning 
    fossil fuels. Carbon dioxide concentrations have increased by 50% since 
    pre-industrial times, reaching 420 ppm in 2023. This has led to a temperature 
    rise of 1.1°C above pre-industrial levels.
    """
    
    result = scorer.score_paragraph(good_para)
    print(f"Good paragraph score: {result['score']}")
    print(f"Breakdown: {result['breakdown']}")
    
    # Bad paragraph
    bad_para = """
    This chapter is organized into three sections. Section 1 discusses background, 
    as examined by Smith et al. (2020), Jones (2019), and Lee et al. (2021). 
    Section 2 presents our methodology.
    """
    
    result = scorer.score_paragraph(bad_para)
    print(f"\nBad paragraph score: {result['score']}")
    print(f"Breakdown: {result['breakdown']}")