"""
Configuration file for Stage 1 pipeline.
Contains all keywords, patterns, and parameters.
"""

# ============================================================================
# CLIMATE KEYWORDS (for identifying useful content)
# ============================================================================

CLIMATE_KEYWORDS = {
    'gases': [
        'co2', 'carbon dioxide', 'methane', 'ch4', 'nitrous oxide', 'n2o',
        'ozone', 'o3', 'greenhouse gas', 'ghg', 'carbon', 'cfc', 'hfc',
        'aerosol', 'sulfur dioxide', 'so2', 'water vapor'
    ],
    
    'effects': [
        'global warming', 'climate change', 'temperature rise', 'warming',
        'sea level rise', 'ocean acidification', 'ice melt', 'glacier',
        'ice sheet', 'permafrost', 'extreme weather', 'heat wave', 'drought',
        'flood', 'storm', 'hurricane', 'cyclone', 'precipitation', 'rainfall'
    ],
    
    'sources': [
        'fossil fuel', 'coal', 'oil', 'petroleum', 'natural gas', 'gasoline',
        'diesel', 'combustion', 'burning', 'deforestation', 'agriculture',
        'livestock', 'cattle', 'rice', 'transportation', 'vehicle', 'car',
        'truck', 'aviation', 'shipping', 'industry', 'cement', 'steel',
        'power plant', 'electricity', 'emission'
    ],
    
    'processes': [
        'greenhouse effect', 'carbon cycle', 'albedo', 'feedback',
        'radiative forcing', 'absorption', 'photosynthesis', 'respiration',
        'evaporation', 'condensation'
    ],
    
    'impacts': [
        'biodiversity', 'ecosystem', 'habitat', 'species', 'extinction',
        'coral reef', 'forest', 'wetland', 'food security', 'water stress',
        'migration', 'health'
    ],
    
    'metrics': [
        'temperature', 'degree', 'celsius', 'fahrenheit', 'ppm',
        'parts per million', 'concentration', 'gigaton', 'metric ton',
        'emissions rate', 'anomaly'
    ],
    
    'policy': [
        'paris agreement', 'ipcc', 'mitigation', 'adaptation',
        'renewable energy', 'solar', 'wind', 'carbon tax',
        'emissions trading', 'net zero', 'carbon neutral', 'sustainability'
    ],
    
    'time': [
        'pre-industrial', 'baseline', 'trend', 'projection',
        'scenario', 'rcp', 'ssp', 'century', 'decade', 'annual'
    ]
}

ALL_CLIMATE_KEYWORDS = []
for category_keywords in CLIMATE_KEYWORDS.values():
    ALL_CLIMATE_KEYWORDS.extend(category_keywords)

# ============================================================================
# CAUSAL LANGUAGE (positive signal)
# ============================================================================

CAUSAL_PHRASES = [
    'causes', 'cause', 'caused by', 'leads to', 'lead to', 'results in',
    'result in', 'produces', 'produce', 'generates', 'generate',
    'contributes to', 'contribute to', 'drives', 'drive', 'triggers',
    'trigger', 'induces', 'induce', 'due to', 'because of',
    'resulting from', 'attributed to', 'stems from', 'arises from',
    'brings about', 'gives rise to'
]

# ============================================================================
# EFFECT/IMPACT LANGUAGE (positive signal)
# ============================================================================

IMPACT_PHRASES = [
    'impacts', 'impact', 'affects', 'affect', 'influences', 'influence',
    'changes', 'change', 'alters', 'alter', 'modifies', 'modify',
    'disrupts', 'disrupt', 'damages', 'damage', 'threatens', 'threaten'
]

# ============================================================================
# FACTUAL LANGUAGE (positive signal)
# ============================================================================

FACTUAL_PHRASES = [
    'studies show', 'research shows', 'evidence indicates', 'data reveal',
    'observations confirm', 'research demonstrates', 'findings show',
    'likely', 'very likely', 'extremely likely', 'virtually certain',
    'high confidence', 'medium confidence'
]

# ============================================================================
# NOISE PATTERNS (negative signals - remove these)
# ============================================================================

META_ORGANIZATIONAL_PHRASES = [
    'this chapter', 'this section', 'this report', 'this paper',
    'this study', 'this document', 'this article', 'as follows',
    'is organized', 'is structured', 'consists of', 'divided into',
    'see section', 'see chapter', 'described in', 'discussed in',
    'presented in', 'shown in', 'section 1', 'section 2', 'section 3',
    'chapter 1', 'chapter 2', 'chapter 3'
]

ACKNOWLEDGMENT_PHRASES = [
    'we thank', 'thank', 'grateful', 'acknowledge', 'acknowledgment',
    'acknowledgement', 'appreciation', 'funding', 'grant', 'support',
    'funded by', 'supported by', 'financial support'
]

METHODOLOGY_PHRASES = [
    'methodology', 'methods section', 'statistical analysis',
    'regression analysis', 'data collection', 'sample size',
    'sampling method', 'survey', 'questionnaire', 'interview',
    'participants', 'protocol', 'procedure'
]

AUTHOR_AFFILIATION_PHRASES = [
    'department of', 'university', 'institute', 'email', '@',
    'corresponding author', 'author contribution', 'orcid'
]

FIGURE_TABLE_PHRASES = [
    'see figure', 'see table', 'shown in figure', 'shown in table',
    'as shown in fig', 'table 1', 'table 2', 'figure 1', 'figure 2',
    'fig.', 'tab.'
]

# ============================================================================
# SCORING WEIGHTS
# ============================================================================

SCORING_WEIGHTS = {
    'climate_keyword_density_high': 30,    # >= 15% keyword density
    'climate_keyword_density_medium': 20,  # >= 10% keyword density
    'climate_keyword_density_low': 10,     # >= 5% keyword density
    'causal_language': 15,
    'quantitative_data': 15,
    'impact_language': 10,
    'factual_tone': 10,
    'proper_length': 5,
    
    'meta_organizational': -20,
    'heavy_citations': -30,
    'acknowledgment': -25,
    'methodology_heavy': -20,
    'author_affiliation': -20,
    'figure_table_only': -15,
    'wrong_length': -10
}

# ============================================================================
# FILTERING PARAMETERS
# ============================================================================

SCORE_THRESHOLD = 50  # Keep paragraphs with score >= 60
MIN_PARAGRAPH_LENGTH = 10  # words
MAX_PARAGRAPH_LENGTH = 500  # words

# ============================================================================
# CHUNKING PARAMETERS
# ============================================================================

CHUNK_SIZE = 500  # characters
CHUNK_OVERLAP = 50  # characters
MIN_CHUNK_SIZE = 100  # minimum chunk size to keep


SECTIONS_TO_REMOVE = [
    'acknowledgment', 'acknowledgement', 'acknowledgments',
    'acknowledgements', 'references', 'bibliography', 'works cited',
    'appendix', 'supplementary', 'author information',
    'author contributions', 'funding', 'competing interests',
    'conflicts of interest', 'table of contents', 'list of figures',
    'list of tables', 'about the author', 'about the authors'
]

# ============================================================================
# TABLE OF CONTENTS / LIST PATTERNS (to remove)
# ============================================================================

TOC_PATTERNS = [
    r'\.{3,}',              # Multiple dots (e.g., "Chapter 1........45")
    r'Box\s+\d+\.\d+:',     # Box references (e.g., "Box 2.1:")
    r'Figure\s+\d+\.\d+:',  # Figure references
    r'Table\s+\d+\.\d+:',   # Table references
    r'Chapter\s+\d+\.{3,}', # Chapter with dots
    r'Section\s+\d+\.{3,}', # Section with dots
]

# Lines that indicate list/TOC structure
LIST_INDICATORS = [
    'list of boxes',
    'list of figures',
    'list of tables',
    'table of contents',
    'contents',
]

# Patterns for page number references
PAGE_REF_PATTERN = r'\.{3,}\s*\d+\s*$'  # Ends with ... and page number