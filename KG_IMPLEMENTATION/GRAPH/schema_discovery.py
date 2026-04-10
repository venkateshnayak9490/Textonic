from neo4j import GraphDatabase

class SchemaDiscovery:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def get_schema(self):
        """Discover all relationships, entity types, and sample entities"""
        
        with self.driver.session() as session:
            rel_result = session.run("CALL db.relationshipTypes()")
            relationships = [r[0] for r in rel_result]
            
            entity_result = session.run("""
                MATCH (n:Entity)
                WHERE n.canonical_label IS NOT NULL
                RETURN DISTINCT n.canonical_label AS label, 
                       collect(n.name)[0..3] AS examples
                ORDER BY label
                LIMIT 20
            """)
            entity_types = [
                {"label": r["label"], "examples": r["examples"]}
                for r in entity_result
            ]
            
            node_count = session.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
            
            relationship_samples = {}
            for rel_type in relationships[:10]:  
                sample = session.run(f"""
                    MATCH (a:Entity)-[r:`{rel_type}`]->(b:Entity)
                    RETURN a.name AS source, b.name AS target
                    LIMIT 3
                """)
                relationship_samples[rel_type] = [
                    {"source": r["source"], "target": r["target"]}
                    for r in sample
                ]
        
        return {
            "relationships": relationships,
            "entity_types": entity_types,
            "node_count": node_count,
            "rel_count": rel_count,
            "relationship_samples": relationship_samples
        }
    
    def format_schema_for_llm(self, schema):
        """Format schema in a readable way for LLM"""
        
        lines = [
            f" **Knowledge Graph Schema**",
            f"",
            f"**Statistics:**",
            f"- Total entities: {schema['node_count']:,}",
            f"- Total relationships: {schema['rel_count']:,}",
            f"",
            f"**Available Relationship Types:**"
        ]
        
        for rel in schema['relationships']:
            lines.append(f"  - {rel}")
            if rel in schema['relationship_samples']:
                for sample in schema['relationship_samples'][rel][:2]:
                    lines.append(f"    Example: '{sample['source']}' -{rel}-> '{sample['target']}'")
        
        lines.append("")
        lines.append("**Entity Types (with examples):**")
        
        for entity_type in schema['entity_types'][:15]: # Limit to 15 to keep prompt short
            examples = ", ".join([f"'{e}'" for e in entity_type['examples'][:3]])
            lines.append(f"  - {entity_type['label']}: {examples}")
        
        return "\n".join(lines)
    
    def close(self):
        self.driver.close()