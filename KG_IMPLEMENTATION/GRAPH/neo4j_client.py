from neo4j import GraphDatabase

class Neo4jClient:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def query(self, cypher):
        with self.driver.session() as session:
            return [r.data() for r in session.run(cypher)]

    def close(self):
        self.driver.close()