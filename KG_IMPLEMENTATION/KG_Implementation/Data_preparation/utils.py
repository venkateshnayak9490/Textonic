"""
Utility functions for text processing.
"""

import re
from typing import List


def clean_text_basic(text: str) -> str:
    if isinstance(text, bytes):
        text = text.decode('utf-8', errors='ignore')
    
    text = text.replace('\x00', '')
    
    text = ''.join(char for char in text if char.isprintable() or char in '\n\t')
    
    return text


def fix_hyphenated_line_breaks(text: str) -> str:
    pattern = r'(\w+)-\s*\n\s*(\w+)'
    text = re.sub(pattern, r'\1\2', text)
    
    return text


def normalize_whitespace(text: str) -> str:
    text = text.replace('\t', ' ')
    
    text = re.sub(r' +', ' ', text)
    
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    lines = text.split('\n')
    lines = [line.strip() for line in lines]
    text = '\n'.join(lines)
    
    return text


def remove_page_numbers(text: str) -> str:
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line_stripped = line.strip()
        if re.match(r'^[\d\s\-]+$', line_stripped):
            continue
        if re.match(r'^Page\s+\d+$', line_stripped, re.IGNORECASE):
            continue
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def split_into_paragraphs(text: str):

    paragraphs = re.split(r'\n\s*\n|\n(?=[A-Z])', text)
    
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    return paragraphs


def count_words(text: str) -> int:

    return len(text.split())


def contains_any_phrase(text: str, phrases: List[str]) -> bool:

    text_lower = text.lower()
    return any(phrase.lower() in text_lower for phrase in phrases)


def count_citations(text: str) -> int:
    pattern1 = r'\([A-Z][a-z]+(?:\s+et\s+al\.)?,\s*\d{4}\)'
    citations1 = re.findall(pattern1, text)
    
    pattern2 = r'\[\d+\]'
    citations2 = re.findall(pattern2, text)
    
    return len(citations1) + len(citations2)


def has_quantitative_data(text: str) -> bool:
    patterns = [
        r'\d+\.?\d*\s*%',           # 50%, 1.5%
        r'\d+\.?\d*\s*°[CF]',       # 1.5°C, 70°F
        r'\d+\.?\d*\s*ppm',         # 400 ppm
        r'\d+\.?\d*\s*gigatons?',   # 10 gigatons
        r'\d+\.?\d*\s*Gt',          # 10 Gt
        r'\d+\.?\d*\s*metric tons?', # 5 metric tons
        r'\d+\.?\d*\s*cm',          # 20 cm
        r'\d+\.?\d*\s*meters?',     # 2 meters
    ]
    
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    
    return False


def remove_email_url_lines(text: str) -> str:

    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        if re.search(r'\S+@\S+\.\S+', line):
            continue
        if re.search(r'https?://\S+', line):
            continue
        if re.search(r'doi:', line, re.IGNORECASE):
            continue
        
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)