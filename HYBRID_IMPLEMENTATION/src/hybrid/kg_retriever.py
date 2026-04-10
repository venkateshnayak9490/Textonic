"""
KG Retriever: Load and query knowledge graph triples and entities.
Supports entity linking and triple retrieval.
"""

import json
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import (
    ENTITIES_JSON, TRIPLES_JSON, CANONICAL_SCHEMA,
    ENTITY_SIMILARITY_THRESHOLD, VERBOSE
)


class KGRetriever:
    """Load and query a knowledge graph from JSON files."""

    def __init__(
        self,
        entities_path: str = None,
        triples_path: str = None,
        schema_path: str = None
    ):
        """
        Initialize KG retriever by loading entities and triples.
        
        Args:
            entities_path: Path to entities.json
            triples_path: Path to triples_2.json
            schema_path: Path to canonical_schema.json
        """
        self.entities_path = entities_path or str(ENTITIES_JSON)
        self.triples_path = triples_path or str(TRIPLES_JSON)
        self.schema_path = schema_path or str(CANONICAL_SCHEMA)
        
        self.entities = []
        self.triples = []
        self.canonical_labels = []
        self.entity_index = {}  # text -> list of entities
        
        self._load_kg()
    
    def _load_kg(self):
        """Load entities, triples, and schema from JSON files."""
        if VERBOSE:
            print(f"Loading KG from {self.entities_path}...")
        
        # Load entities
        try:
            with open(self.entities_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.entities = data.get('entities', data) if isinstance(data, dict) else data
            if VERBOSE:
                print(f"  Loaded {len(self.entities)} entities")
        except Exception as e:
            print(f"Error loading entities: {e}")
            self.entities = []
        
        # Load triples
        try:
            with open(self.triples_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.triples = data if isinstance(data, list) else data.get('triples', [])
            if VERBOSE:
                print(f"  Loaded {len(self.triples)} triples")
        except Exception as e:
            print(f"Error loading triples: {e}")
            self.triples = []
        
        # Load schema
        try:
            with open(self.schema_path, 'r', encoding='utf-8') as f:
                schema = json.load(f)
                self.canonical_labels = schema.get('canonical_labels', [])
            if VERBOSE:
                print(f"  Loaded {len(self.canonical_labels)} canonical labels")
        except Exception as e:
            print(f"Error loading schema: {e}")
            self.canonical_labels = []
        
        # Build entity index
        self._build_entity_index()
    
    def _build_entity_index(self):
        """Build a quick lookup index: entity_text -> list of entity records."""
        for entity in self.entities:
            text = entity.get('text', '').lower()
            if text:
                if text not in self.entity_index:
                    self.entity_index[text] = []
                self.entity_index[text].append(entity)
    
    def retrieve_entities_by_text(
        self,
        query_text: str,
        top_k: int = 5,
        exact_match: bool = False
    ) -> List[Dict]:
        """
        Retrieve entities that match query text.
        
        Args:
            query_text: Text fragment to match (e.g., "greenhouse gas", "fossil fuels")
            top_k: Number of results to return
            exact_match: If True, only exact matches; if False, substring matches
        
        Returns:
            List of entity dicts with text, canonical_label, source_file
        """
        # Validate input
        if not query_text or not isinstance(query_text, str):
            return []
        
        query_lower = query_text.strip().lower()
        if not query_lower:
            return []
        
        results = []
        
        for entity_text, entities_list in self.entity_index.items():
            if exact_match:
                match = entity_text == query_lower
            else:
                match = query_lower in entity_text or entity_text in query_lower
            
            if match:
                results.extend(entities_list)
        
        # Sort by frequency/confidence
        results = results[:top_k]
        return results
    
    def retrieve_triples_from_entity(
        self,
        entity_text: str,
        hops: int = 1,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Retrieve triples where entity is subject or object.
        
        Args:
            entity_text: Entity text to search for
            hops: Number of relationship hops (1 = direct relations, 2 = indirect)
            top_k: Maximum results to return
        
        Returns:
            List of triples
        """
        entity_text_lower = entity_text.lower()
        results = []
        
        # Direct matches
        for triple in self.triples:
            subj = triple.get('subject', '').lower()
            obj = triple.get('object', '').lower()
            
            if entity_text_lower in subj or entity_text_lower in obj:
                results.append(triple)
        
        # Remove duplicates
        seen = set()
        deduped = []
        for t in results:
            key = (t.get('subject'), t.get('relation'), t.get('object'))
            if key not in seen:
                seen.add(key)
                deduped.append(t)
        
        return deduped[:top_k]
    
    def retrieve_by_label(
        self,
        label: str,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Retrieve all entities with a given canonical label.
        
        Args:
            label: Canonical label (e.g., "greenhouse_gas", "emission_source")
            top_k: Max results to return
        
        Returns:
            List of entity dicts
        """
        results = [
            e for e in self.entities
            if e.get('canonical_label') == label
        ]
        return results[:top_k]
    
    def retrieve_triples_by_relation(
        self,
        relation: str,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Retrieve all triples with a given relation type.
        
        Args:
            relation: Relation name (e.g., "EMITS", "CAUSES", "MITIGATES")
            top_k: Max results to return
        
        Returns:
            List of triple dicts
        """
        results = [
            t for t in self.triples
            if t.get('relation') == relation
        ]
        return results[:top_k]
    
    def query_kg(
        self,
        query: str,
        top_k: int = 10,
        use_labels: bool = False
    ) -> Dict:
        """
        General KG query: retrieve relevant entities and triples.
        
        Args:
            query: Natural language question or search term
            top_k: Number of results to return
            use_labels: If True, try to match canonical labels in query
        
        Returns:
            Dict with 'entities' and 'triples' lists
        """
        # Validate input
        if not query or not isinstance(query, str) or not query.strip():
            return {'entities': [], 'triples': [], 'num_entities': 0, 'num_triples': 0}
        
        entities = self.retrieve_entities_by_text(query, top_k=top_k)
        
        # Also get triples from retrieved entities
        triples_set = []
        for entity in entities:
            triples = self.retrieve_triples_from_entity(
                entity.get('text', ''),
                hops=1,
                top_k=top_k
            )
            triples_set.extend(triples)
        
        # Deduplicate triples
        seen = set()
        deduped_triples = []
        for t in triples_set:
            key = (t.get('subject'), t.get('relation'), t.get('object'))
            if key not in seen:
                seen.add(key)
                deduped_triples.append(t)
        
        return {
            'entities': entities,
            'triples': deduped_triples[:top_k],
            'num_entities': len(entities),
            'num_triples': len(deduped_triples),
        }
    
    def format_kg_context(self, kg_results: Dict) -> str:
        """
        Format KG results into a readable context string for LLM.
        
        Args:
            kg_results: Output from query_kg()
        
        Returns:
            Formatted text for insertion into prompt
        """
        context_parts = []
        
        # Format entities with null-safe handling
        if kg_results.get('entities'):
            context_parts.append("**Entities:**")
            for entity in kg_results['entities']:
                text = entity.get('text', 'UNKNOWN')
                label = entity.get('canonical_label', 'UNKNOWN')
                source = entity.get('source_file', 'UNKNOWN')
                # Skip if all fields are missing
                if text != 'UNKNOWN' or label != 'UNKNOWN':
                    context_parts.append(f"  - {text} ({label})")
        
        # Format triples with null-safe handling
        if kg_results.get('triples'):
            context_parts.append("\n**Relationships:**")
            for triple in kg_results['triples']:
                subj = triple.get('subject', 'UNKNOWN')
                rel = triple.get('relation', 'UNKNOWN')
                obj = triple.get('object', 'UNKNOWN')
                # Skip malformed triples
                if subj != 'UNKNOWN' and rel != 'UNKNOWN' and obj != 'UNKNOWN':
                    context_parts.append(f"  - {subj} --[{rel}]--> {obj}")
        
        return "\n".join(context_parts) if context_parts else "No knowledge graph matches."


# Example usage
if __name__ == "__main__":
    retriever = KGRetriever()
    
    # Test query
    results = retriever.query_kg("fossil fuel emissions", top_k=5)
    print("\nKG Query Results:")
    print(f"Entities: {len(results['entities'])}")
    print(f"Triples: {len(results['triples'])}")
    
    context = retriever.format_kg_context(results)
    print("\nFormatted Context:")
    print(context)
