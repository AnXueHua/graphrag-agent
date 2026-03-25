import pandas as pd
from neo4j import GraphDatabase
import os
from config import Config
import numpy as np

'''
graphrag数据转入neo4j
'''
OUTPUT_DIR = Config.GRAPHRAG_OUTPUT_DIR
LANCEDB_URI = Config.GRAPHRAG_LANCEDB_DIR
NEO4J_URI = Config.NEO4J_URI
NEO4J_USERNAME = Config.NEO4J_USERNAME
NEO4J_PASSWORD = Config.NEO4J_PASSWORD
NEO4J_DATABASE = Config.NEO4J_DATABASE

class GraphRAG2Neo4j:
    def __init__(self, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    def close(self):
        self.driver.close()

    def clear_database(self):
        print("🧹 正在清空数据库 (删除所有旧数据)...")
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("   数据库已清空。")

    def create_index(self):
        print("正在创建索引以加速导入...")
        with self.driver.session() as session:
            # 实体索引
            session.run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE")
            session.run("CREATE INDEX entity_name IF NOT EXISTS FOR (n:Entity) ON (n.name)")
            # 社区索引
            session.run("CREATE CONSTRAINT community_id IF NOT EXISTS FOR (c:Community) REQUIRE c.id IS UNIQUE")
            # 文本块索引
            session.run("CREATE CONSTRAINT textunit_id IF NOT EXISTS FOR (t:TextUnit) REQUIRE t.id IS UNIQUE")
            # 文档索引
            session.run("CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE")
            # 如果要使用向量索引 (需要 Neo4j 5.x+)
            try:
                session.run("""
                    CREATE VECTOR INDEX entity_embedding IF NOT EXISTS
                    FOR (n:Entity) ON (n.embedding)
                    OPTIONS {indexConfig: {
                     `vector.dimensions`: 1536,
                     `vector.similarity_function`: 'cosine'
                    }}
                """)
            except Exception as e:
                print(f"跳过向量索引创建: {e}")

    def batch_run(self, cypher, data, batch_size=1000):
        total = len(data)
        with self.driver.session() as session:
            for i in range(0, total, batch_size):
                batch = data[i:i+batch_size]
                session.run(cypher, rows=batch)
                print(f"   进度: {min(i+batch_size, total)} / {total}")

# ==================== 1. 导入文档 (Documents) ====================
    def import_documents(self):
        print("\n--- 1. 导入文档 (Documents) ---")
        df = pd.read_parquet(os.path.join(OUTPUT_DIR, "documents.parquet"))
        rows = df.fillna("").to_dict('records')
        
        cypher = """
        UNWIND $rows AS row
        MERGE (d:Document {id: row.id})
        SET d.title = row.title,
            d.text = row.text,
            d.creation_date = toString(row.creation_date)
        """
        self.batch_run(cypher, rows)

    # ==================== 2. 导入文本块 (TextUnits) ====================
    def import_text_units(self):
        print("\n--- 2. 导入文本块 (TextUnits) ---")
        # 读取基础信息
        units_df = pd.read_parquet(os.path.join(OUTPUT_DIR, "text_units.parquet"))
        # 读取向量信息
        emb_df = pd.read_parquet(os.path.join(OUTPUT_DIR, "embeddings.text_unit.text.parquet"))
        # 合并 
        df = units_df.merge(emb_df, on="id", how="left")
        rows = df.fillna("").to_dict('records')
        cypher = """
        UNWIND $rows AS row
        MERGE (t:TextUnit {id: row.id})
        SET t.text = row.text,
            t.n_tokens = row.n_tokens
        
        FOREACH (_ IN CASE WHEN row.embedding IS NOT NULL THEN [1] ELSE [] END |
            SET t.embedding = row.embedding
        )
        
        FOREACH (doc_id IN row.document_ids | 
            MERGE (d:Document {id: doc_id})
            MERGE (d)-[:HAS_PART]->(t)
        )
        """
        self.batch_run(cypher, rows)

    # ==================== 3. 导入实体 (Entities) ====================
    def import_entities(self):
        print("\n--- 3. 导入实体 (Entities) ---")
        entities_df = pd.read_parquet(os.path.join(OUTPUT_DIR, "entities.parquet"))
        emb_df = pd.read_parquet(os.path.join(OUTPUT_DIR, "embeddings.entity.description.parquet"))
        
        # 合并实体属性与向量
        df = entities_df.merge(emb_df, on="id", how="left")
        
        rows = df.fillna("").to_dict('records')
        cypher = """
        UNWIND $rows AS row
        MERGE (e:Entity {id: row.id})
        SET e.name = row.title,
            e.type = row.type,
            e.description = row.description,
            e.human_readable_id = row.human_readable_id,
            e.x = row.x,
            e.y = row.y
        
        FOREACH (_ IN CASE WHEN row.embedding IS NOT NULL THEN [1] ELSE [] END |
            SET e.embedding = row.embedding
        )
        
        FOREACH (tu_id IN row.text_unit_ids | 
            MERGE (t:TextUnit {id: tu_id})
            MERGE (t)-[:MENTIONS]->(e)
        )
        """
        self.batch_run(cypher, rows)

    # ==================== 4. 导入关系 (Relationships) ====================
    def import_relationships(self):
        print("\n--- 4. 导入关系 (Relationships) ---")
        df = pd.read_parquet(os.path.join(OUTPUT_DIR, "relationships.parquet"))
        rows = df.fillna("").to_dict('records')
        
        cypher = """
        UNWIND $rows AS row
        MATCH (source:Entity {name: row.source})
        MATCH (target:Entity {name: row.target})
        MERGE (source)-[r:RELATED]->(target)
        SET r.weight = row.weight,
            r.description = row.description,
            r.combined_degree = row.combined_degree
        """
        self.batch_run(cypher, rows)

    # ==================== 5. 导入社区 (Communities) ====================
    def import_communities(self):
        print("\n--- 5. 导入社区 (Communities) ---")
        report_df = pd.read_parquet(os.path.join(OUTPUT_DIR, "community_reports.parquet"))
        # 社区向量
        emb_df = pd.read_parquet(os.path.join(OUTPUT_DIR, "embeddings.community.full_content.parquet"))
        report_df = report_df.merge(emb_df, on="id", how="left")
        rows = report_df.fillna("").to_dict('records')
        cypher = """
        UNWIND $rows AS row
        MERGE (c:Community {community: row.community})
        SET c.title = row.title,
            c.summary = row.summary,
            c.full_content = row.full_content,
            c.rank = row.rank,
            c.level = row.level,
            c.uuid = row.id
        
        FOREACH (_ IN CASE WHEN row.embedding IS NOT NULL THEN [1] ELSE [] END |
            SET c.embedding = row.embedding
        )
        """
        self.batch_run(cypher, rows)

        # 建立实体-社区关系
        struct_df = pd.read_parquet(os.path.join(OUTPUT_DIR, "communities.parquet"))
        struct_df = struct_df[['community', 'entity_ids']].dropna()
        rows = struct_df.to_dict('records')
        cypher_rel = """
        UNWIND $rows AS row
        MERGE (c:Community {community: row.community})
        
        FOREACH (e_id IN row.entity_ids | 
            MERGE (e:Entity {id: e_id})
            MERGE (e)-[:IN_COMMUNITY]->(c)
        )
        """
        self.batch_run(cypher_rel, rows)

if __name__ == "__main__":
    importer = GraphRAG2Neo4j(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
    
    try:
        importer.create_index()
        importer.import_documents()     # 导入文档
        importer.import_text_units()    # 导入文本块（含向量）
        importer.import_entities()      # 导入实体（含向量）
        importer.import_relationships() # 导入关系
        importer.import_communities()   # 导入社区（含向量）
        print("\n=== 所有数据导入完成！===")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        importer.close()