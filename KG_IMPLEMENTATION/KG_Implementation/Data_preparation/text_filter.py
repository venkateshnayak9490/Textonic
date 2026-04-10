"""
Content-based text filtering.
Filters paragraphs based on scores.
"""

from typing import List, Dict, Tuple
from content_scorer import ContentScorer
from utils import (
    clean_text_basic, fix_hyphenated_line_breaks, normalize_whitespace,
    remove_page_numbers, split_into_paragraphs, remove_email_url_lines
)
from config import SCORE_THRESHOLD
import re

def remove_toc_noise(text: str) -> str:
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line_strip = line.strip()
        
        if re.search(r'\.{5,}', line_strip):
            continue
        
        if re.match(r'^(Box|Figure|Table)\s+\d', line_strip):
            continue
        
        if line_strip.lower().startswith('list of'):
            continue
        
        if len(line_strip.split()) < 5:
            continue
        
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

class TextFilter:

    def __init__(self, score_threshold: int = SCORE_THRESHOLD):
        self.scorer = ContentScorer()
        self.score_threshold = score_threshold
    
    def filter_text(self, raw_text: str, verbose: bool = True) -> Dict:

        if verbose:
            print("Starting text filtering...")
        
        # Step 1: Basic cleaning
        if verbose:
            print("  Step 1/6: Basic cleaning...")
        text = clean_text_basic(raw_text)
        text = fix_hyphenated_line_breaks(text)
        text = normalize_whitespace(text)
        text = remove_page_numbers(text)
        text = remove_email_url_lines(text)
        text = remove_toc_noise(text)
        
        if verbose:
            print("  Step 2/6: Splitting into paragraphs...")
        paragraphs = split_into_paragraphs(text)
        
        if verbose:
            print(f"    Found {len(paragraphs)} paragraphs")
        
        if verbose:
            print("  Step 3/6: Scoring paragraphs...")
        
        scored_paragraphs = []
        for para in paragraphs:
            score_result = self.scorer.score_paragraph(para)
            scored_paragraphs.append({
                'text': para,
                'score': score_result['score'],
                'word_count': score_result['word_count'],
                'keyword_density': score_result['keyword_density']
            })
        
        if verbose:
            print(f"  Step 4/6: Filtering (threshold={self.score_threshold})...")
        
        kept_paragraphs = [
            p for p in scored_paragraphs 
            if p['score'] >= self.score_threshold
        ]
        
        if verbose:
            print(f"    Kept {len(kept_paragraphs)}/{len(paragraphs)} paragraphs")
        
        if verbose:
            print("  Step 5/6: Reconstructing text...")
        
        filtered_text = '\n\n'.join(p['text'] for p in kept_paragraphs)
        
        if verbose:
            print("  Step 6/6: Final cleanup...")
        
        filtered_text = normalize_whitespace(filtered_text)
        
        stats = {
            'original_length': len(raw_text),
            'filtered_length': len(filtered_text),
            'retention_rate': len(filtered_text) / len(raw_text) if len(raw_text) > 0 else 0,
            'original_paragraphs': len(paragraphs),
            'kept_paragraphs': len(kept_paragraphs),
            'removed_paragraphs': len(paragraphs) - len(kept_paragraphs),
            'avg_score_kept': sum(p['score'] for p in kept_paragraphs) / len(kept_paragraphs) if kept_paragraphs else 0,
            'avg_keyword_density': sum(p['keyword_density'] for p in kept_paragraphs) / len(kept_paragraphs) if kept_paragraphs else 0
        }
        
        if verbose:
            print(f"\nFiltering complete:")
            print(f"  Original: {stats['original_length']:,} chars, {stats['original_paragraphs']} paragraphs")
            print(f"  Filtered: {stats['filtered_length']:,} chars, {stats['kept_paragraphs']} paragraphs")
            print(f"  Retention: {stats['retention_rate']:.1%}")
            print(f"  Avg score: {stats['avg_score_kept']:.1f}")
            print(f"  Avg keyword density: {stats['avg_keyword_density']:.2%}")
        
        return {
            'filtered_text': filtered_text,
            'statistics': stats,
            'scored_paragraphs': scored_paragraphs,
            'kept_paragraphs': kept_paragraphs
        }

# Example usage
if __name__ == '__main__':
    # Sample text
    sample_text = """
    Climate Change Report
    
    John Smith¹, Jane Doe²
    ¹MIT, ²Stanford University
    
    Abstract
    
    Global warming is primarily caused by greenhouse gas emissions from burning 
    fossil fuels. Carbon dioxide concentrations have increased by 50% since 
    pre-industrial times, reaching 420 ppm in 2023.
    
    This chapter is organized into three sections. Section 1 discusses background.
    
    The greenhouse effect occurs when gases like CO2 and methane trap heat in 
    the atmosphere, leading to rising global temperatures.
    
    Acknowledgments
    
    We thank Dr. Brown for helpful comments.
    """
    
    filter = TextFilter()
    result = filter.filter_text(sample_text)
    
    print("\n" + "="*60)
    print("FILTERED TEXT:")
    print("="*60)
    print(result['filtered_text'])