import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from neo4j import GraphDatabase

from config import Config


"""
GraphRAG 3.1.2 -> Neo4j 数据导入工具

设计目标
--------
1. 兼容 GraphRAG 3.1.2 最终生成的 Parquet 输出文件。
2. 支持增量/幂等导入，不清空整个 Neo4j 数据库。
3. 通过 kb_id 实现多个知识库之间的数据隔离。
4. 保留原始文本证据链：
      Document -[:HAS_PART]-> TextUnit -[:MENTIONS]-> Entity
      Entity   -[:RELATED]->  Entity
      Entity   -[:IN_COMMUNITY]-> Community
5. 可选支持将 GraphRAG LanceDB 中的向量迁移为 Neo4j 节点向量属性。
6. 在写入 Neo4j 前校验数据结构、实体类型、孤立关系以及证据引用完整性。

说明
----
GraphRAG 3.x 默认不再将向量字段或独立向量文件导出到 Parquet。
向量会直接写入配置的向量数据库，本项目中使用 LanceDB。
因此本导入工具不再依赖以下旧版文件：
    embeddings.text_unit.text.parquet
    embeddings.entity.description.parquet
    embeddings.community.full_content.parquet
"""


OUTPUT_DIR = Path(Config.GRAPHRAG_OUTPUT_DIR)
LANCEDB_URI = Path(Config.GRAPHRAG_LANCEDB_DIR)

NEO4J_URI = Config.NEO4J_URI
NEO4J_USERNAME = Config.NEO4J_USERNAME
NEO4J_PASSWORD = Config.NEO4J_PASSWORD
NEO4J_DATABASE = getattr(Config, "NEO4J_DATABASE", "neo4j")

# 可选配置项
KB_ID = getattr(Config, "GRAPHRAG_KB_ID", OUTPUT_DIR.name)
BATCH_SIZE = int(getattr(Config, "NEO4J_BATCH_SIZE", 500))
IMPORT_VECTORS = bool(getattr(Config, "GRAPHRAG_IMPORT_VECTORS", False))
SYNC_DELETE_STALE = bool(getattr(Config, "GRAPHRAG_SYNC_DELETE_STALE", False))
VECTOR_TABLE_NAME = getattr(Config, "GRAPHRAG_VECTOR_TABLE", "vector_index")


ALLOWED_ENTITY_TYPES = {
    "设备",
    "设备部件",
    "组织机构",
    "人员角色",
    "标准规范",
    "技术文档",
    "文件记录",
    "检测试验项目",
    "检测方法",
    "检查方式",
    "检测仪器",
    "技术参数",
    "判定标准",
    "验收环节",
    "巡视类型",
    "检修类别",
    "检修工艺",
    "故障异常",
    "材料介质",
    "地点设施",
}


CORE_FILES = {
    "documents": "documents.parquet",
    "text_units": "text_units.parquet",
    "entities": "entities.parquet",
    "relationships": "relationships.parquet",
}

OPTIONAL_FILES = {
    "communities": "communities.parquet",
    "community_reports": "community_reports.parquet",
}


def _is_missing(value: Any) -> bool:
    """仅当输入是标量 NA、NaN 或 None 时返回 True。"""
    if value is None:
        return True
    if isinstance(value, (list, tuple, set, dict, np.ndarray)):
        return False
    try:
        missing = pd.isna(value)
        return bool(missing) if isinstance(missing, (bool, np.bool_)) else False
    except Exception:
        return False


def _neo4j_value(value: Any) -> Any:
    """
    将 pandas/numpy 数据转换为 Neo4j 可接受的属性值。

    Neo4j 属性支持标量和同类型列表，但不支持任意字典结构。
    字典类型统一编码为 JSON 字符串后保存。
    """
    if _is_missing(value):
        return None

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, (list, tuple, set)):
        result = []
        for item in value:
            item = _neo4j_value(item)
            if item is not None:
                result.append(item)
        return result

    if isinstance(value, dict):
        clean = {str(k): _neo4j_value(v) for k, v in value.items()}
        return json.dumps(clean, ensure_ascii=False)

    if isinstance(value, (str, int, float, bool)):
        return value

    # 对 UUID、Path、Decimal、日期时间类等其他对象统一回退为字符串。
    return str(value)


def _id_list(value: Any) -> list[str]:
    """将 GraphRAG 中列表形式的 ID 字段统一规范为字符串列表。"""
    if _is_missing(value):
        return []

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if not isinstance(value, (list, tuple, set)):
        value = [value]

    result: list[str] = []
    for item in value:
        if not _is_missing(item):
            result.append(str(_neo4j_value(item)))
    return result


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _textunit_document_ids(row: Any) -> list[str]:
    """
    兼容不同 GraphRAG 版本的 TextUnit -> Document 字段。

    GraphRAG 3.1.2 使用单值字段 document_id；
    部分旧版可能使用列表字段 document_ids。
    """
    if hasattr(row, "index") and "document_ids" in row.index:
        return _id_list(row.get("document_ids"))

    if hasattr(row, "index") and "document_id" in row.index:
        return _id_list(row.get("document_id"))

    if isinstance(row, dict):
        if "document_ids" in row:
            return _id_list(row.get("document_ids"))
        if "document_id" in row:
            return _id_list(row.get("document_id"))

    return []

class GraphRAG2Neo4j:
    def __init__(
        self,
        neo4j_uri: str,
        neo4j_username: str,
        neo4j_password: str,
        database: str = "neo4j",
        output_dir: Path | str = OUTPUT_DIR,
        lancedb_uri: Path | str = LANCEDB_URI,
        kb_id: str = KB_ID,
        batch_size: int = BATCH_SIZE,
        import_vectors: bool = IMPORT_VECTORS,
        sync_delete_stale: bool = SYNC_DELETE_STALE,
    ):
        self.database = database
        self.output_dir = Path(output_dir)
        self.lancedb_uri = Path(lancedb_uri)
        self.kb_id = str(kb_id)
        self.batch_size = int(batch_size)
        self.import_vectors = bool(import_vectors)
        self.sync_delete_stale = bool(sync_delete_stale)

        self.run_id = uuid.uuid4().hex
        self.imported_at = _utc_now()

        self.driver = GraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_username, neo4j_password),
        )
        self.driver.verify_connectivity()

        self.tables: dict[str, pd.DataFrame] = {}
        self.vector_map: dict[str, list[float]] = {}
        self.vector_dim: int | None = None

    # ------------------------------------------------------------------
    # 生命周期管理
    # ------------------------------------------------------------------

    def close(self):
        self.driver.close()

    def _session(self):
        return self.driver.session(database=self.database)

    # ------------------------------------------------------------------
    # 输入数据加载与校验
    # ------------------------------------------------------------------

    def load_tables(self):
        print("\n=== 读取 GraphRAG 输出 ===")
        print(f"知识库 ID: {self.kb_id}")
        print(f"输出目录 : {self.output_dir}")

        for name, filename in CORE_FILES.items():
            path = self.output_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"缺少 GraphRAG 核心文件: {path}")
            df = pd.read_parquet(path)
            self.tables[name] = df
            print(f"  {filename:<30} {len(df):>8} rows")

        for name, filename in OPTIONAL_FILES.items():
            path = self.output_dir / filename
            if path.exists():
                df = pd.read_parquet(path)
                self.tables[name] = df
                print(f"  {filename:<30} {len(df):>8} rows")
            else:
                print(f"  {filename:<30} [可选文件不存在，跳过]")

        print("\n=== Parquet Schema ===")
        for name, df in self.tables.items():
            print(f"[{name}] {df.columns.tolist()}")

    @staticmethod
    def _require_columns(df: pd.DataFrame, table: str, columns: Iterable[str]):
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"{table}.parquet 缺少必要字段: {missing}; "
                f"当前字段: {df.columns.tolist()}"
            )

    def validate(self):
        print("\n=== 导入前校验 ===")

        docs = self.tables["documents"]
        text_units = self.tables["text_units"]
        entities = self.tables["entities"]
        rels = self.tables["relationships"]

        self._require_columns(docs, "documents", ["id", "title", "text"])
        self._require_columns(
            text_units,
            "text_units",
            ["id", "text"],
        )
        # GraphRAG 3.1.2 使用 document_id；
        # 旧版本使用 document_ids。
        if (
            "document_id" not in text_units.columns
            and "document_ids" not in text_units.columns
        ):
            raise ValueError(
                "text_units.parquet 缺少文档关联字段："
                "需要 document_id 或 document_ids 其中之一；"
                f"当前字段: {text_units.columns.tolist()}"
            )
        
        self._require_columns(
            entities,
            "entities",
            ["id", "title", "type", "description", "text_unit_ids"],
        )
        self._require_columns(
            rels,
            "relationships",
            ["id", "source", "target", "description", "weight"],
        )

        # GraphRAG 最终关系表通过实体标题解析关系两端的实体。
        title_series = entities["title"].astype(str)
        duplicate_titles = (
            title_series[title_series.duplicated(keep=False)]
            .value_counts()
            .sort_values(ascending=False)
        )
        if len(duplicate_titles):
            preview = duplicate_titles.head(20).to_dict()
            raise ValueError(
                "entities.parquet 存在重复 title，关系端点将产生歧义。"
                f"示例: {preview}"
            )

        entity_titles = set(title_series)
        orphan_mask = (
            ~rels["source"].astype(str).isin(entity_titles)
            | ~rels["target"].astype(str).isin(entity_titles)
        )
        orphan_count = int(orphan_mask.sum())

        self_loop_count = int(
            (rels["source"].astype(str) == rels["target"].astype(str)).sum()
        )

        entity_types = set(
            entities["type"].dropna().astype(str).str.strip().tolist()
        )
        invalid_types = sorted(entity_types - ALLOWED_ENTITY_TYPES)

        # 证据引用完整性检查
        text_unit_ids = set(text_units["id"].astype(str))
        document_ids = set(docs["id"].astype(str))

        unknown_doc_refs = 0
        for _, row in text_units.iterrows():
            unknown_doc_refs += sum(
                doc_id not in document_ids
                for doc_id in _textunit_document_ids(row)
            )

        unknown_entity_text_refs = 0
        for value in entities["text_unit_ids"]:
            unknown_entity_text_refs += sum(
                tu_id not in text_unit_ids for tu_id in _id_list(value)
            )

        unknown_rel_text_refs = 0
        if "text_unit_ids" in rels.columns:
            for value in rels["text_unit_ids"]:
                unknown_rel_text_refs += sum(
                    tu_id not in text_unit_ids for tu_id in _id_list(value)
                )

        print(f"documents             : {len(docs)}")
        print(f"text_units            : {len(text_units)}")
        print(f"entities              : {len(entities)}")
        print(f"relationships         : {len(rels)}")
        print(f"orphan relationships  : {orphan_count}")
        print(f"self loops            : {self_loop_count}")
        print(f"invalid entity types  : {len(invalid_types)}")
        print(f"unknown document refs : {unknown_doc_refs}")
        print(f"unknown entity TU refs: {unknown_entity_text_refs}")
        print(f"unknown rel TU refs   : {unknown_rel_text_refs}")

        if invalid_types:
            counts = (
                entities.loc[
                    ~entities["type"].astype(str).isin(ALLOWED_ENTITY_TYPES),
                    "type",
                ]
                .value_counts()
                .to_dict()
            )
            print(f"⚠ 非 frozen taxonomy 类型: {counts}")

        if orphan_count:
            preview = rels.loc[
                orphan_mask,
                ["source", "target", "description"],
            ].head(10)
            print("⚠ 存在 orphan relationship，导入时这些边将无法 MATCH:")
            print(preview.to_string(index=False))

        # 在生产导入中，孤立关系属于图结构完整性错误。
        if orphan_count:
            raise ValueError(
                f"发现 {orphan_count} 条 orphan relationships，请先修复或过滤。"
            )

    # ------------------------------------------------------------------
    # LanceDB 向量数据
    # ------------------------------------------------------------------

    def load_vectors(self):
        """
        从 GraphRAG 的 LanceDB 数据表中读取 id -> vector 映射。

        GraphRAG 3.x 会直接把向量写入向量数据库，
        不再额外导出独立的 embedding Parquet 文件。
        """
        if not self.import_vectors:
            print("\n=== 向量导入已关闭，跳过 LanceDB ===")
            return

        print("\n=== 读取 LanceDB 向量 ===")
        try:
            import lancedb
        except ImportError:
            print("⚠ 未安装 lancedb，跳过向量导入。")
            self.import_vectors = False
            return

        if not self.lancedb_uri.exists():
            print(f"⚠ LanceDB 目录不存在: {self.lancedb_uri}")
            self.import_vectors = False
            return

        db = lancedb.connect(str(self.lancedb_uri))

        # 兼容不同版本 LanceDB 的表枚举接口。
        table_names = []
        try:
            table_names = list(db.table_names())
        except Exception:
            try:
                result = db.list_tables()
                if isinstance(result, list):
                    table_names = [
                        x if isinstance(x, str) else getattr(x, "name", str(x))
                        for x in result
                    ]
                elif hasattr(result, "tables"):
                    table_names = list(result.tables)
            except Exception:
                pass

        print(f"LanceDB tables: {table_names or '[无法枚举]'}")

        try:
            table = db.open_table(VECTOR_TABLE_NAME)
        except Exception as e:
            print(
                f"⚠ 无法打开向量表 '{VECTOR_TABLE_NAME}': {e}; "
                "跳过向量导入。"
            )
            self.import_vectors = False
            return

        vec_df = table.to_pandas()
        if "id" not in vec_df.columns or "vector" not in vec_df.columns:
            print(
                "⚠ LanceDB 表缺少 id/vector 字段，"
                f"当前字段: {vec_df.columns.tolist()}"
            )
            self.import_vectors = False
            return

        duplicate_ids = int(vec_df["id"].astype(str).duplicated().sum())
        if duplicate_ids:
            print(
                f"⚠ LanceDB 存在 {duplicate_ids} 个重复 id；"
                "将保留最后一条。"
            )

        vector_map: dict[str, list[float]] = {}
        dims: set[int] = set()

        for _, row in vec_df.iterrows():
            vec = row["vector"]
            if _is_missing(vec):
                continue
            if isinstance(vec, np.ndarray):
                vec = vec.tolist()
            else:
                vec = list(vec)

            clean_vec = [float(x) for x in vec]
            vector_map[str(row["id"])] = clean_vec
            dims.add(len(clean_vec))

        if not vector_map:
            print("⚠ LanceDB 中没有可用向量。")
            self.import_vectors = False
            return

        if len(dims) != 1:
            raise ValueError(f"LanceDB 中存在不同向量维度: {sorted(dims)}")

        self.vector_map = vector_map
        self.vector_dim = next(iter(dims))

        print(f"向量数量: {len(self.vector_map)}")
        print(f"向量维度: {self.vector_dim}")

        # 统计各类 GraphRAG 数据与向量 ID 的匹配情况；当多个逻辑向量空间
        # 共用同一个 LanceDB 物理表时，该信息尤其有助于排查问题。
        for name in ("entities", "text_units", "community_reports"):
            if name not in self.tables:
                continue
            df = self.tables[name]
            matched = int(df["id"].astype(str).isin(self.vector_map).sum())
            print(f"{name:<20}: {matched}/{len(df)} matched vectors")

    # ------------------------------------------------------------------
    # Neo4j 数据结构与索引
    # ------------------------------------------------------------------

    def create_schema(self):
        print("\n=== 创建 Neo4j 约束/索引 ===")
        statements = [
            "CREATE CONSTRAINT entity_uid IF NOT EXISTS "
            "FOR (n:Entity) REQUIRE n.uid IS UNIQUE",

            "CREATE CONSTRAINT document_uid IF NOT EXISTS "
            "FOR (n:Document) REQUIRE n.uid IS UNIQUE",

            "CREATE CONSTRAINT textunit_uid IF NOT EXISTS "
            "FOR (n:TextUnit) REQUIRE n.uid IS UNIQUE",

            "CREATE CONSTRAINT community_key IF NOT EXISTS "
            "FOR (n:Community) REQUIRE n.community_key IS UNIQUE",

            "CREATE INDEX entity_lookup IF NOT EXISTS "
            "FOR (n:Entity) ON (n.kb_id, n.name)",

            "CREATE INDEX entity_type IF NOT EXISTS "
            "FOR (n:Entity) ON (n.kb_id, n.type)",

            "CREATE INDEX document_kb IF NOT EXISTS "
            "FOR (n:Document) ON (n.kb_id)",

            "CREATE INDEX textunit_kb IF NOT EXISTS "
            "FOR (n:TextUnit) ON (n.kb_id)",

            "CREATE INDEX community_kb IF NOT EXISTS "
            "FOR (n:Community) ON (n.kb_id)",
        ]

        with self._session() as session:
            for stmt in statements:
                session.run(stmt).consume()

        if self.import_vectors and self.vector_dim:
            self._create_vector_indexes(self.vector_dim)

    def _create_vector_indexes(self, dim: int):
        print(f"创建 Neo4j Vector Index，dimension={dim} ...")
        statements = [
            f"""
            CREATE VECTOR INDEX entity_embedding IF NOT EXISTS
            FOR (n:Entity) ON (n.embedding)
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {dim},
                `vector.similarity_function`: 'cosine'
            }}}}
            """,
            f"""
            CREATE VECTOR INDEX textunit_embedding IF NOT EXISTS
            FOR (n:TextUnit) ON (n.embedding)
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {dim},
                `vector.similarity_function`: 'cosine'
            }}}}
            """,
            f"""
            CREATE VECTOR INDEX community_embedding IF NOT EXISTS
            FOR (n:Community) ON (n.embedding)
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {dim},
                `vector.similarity_function`: 'cosine'
            }}}}
            """,
        ]

        with self._session() as session:
            for stmt in statements:
                try:
                    session.run(stmt).consume()
                except Exception as e:
                    print(f"⚠ 创建向量索引失败，继续导入: {e}")

    # ------------------------------------------------------------------
    # 通用批量写入逻辑
    # ------------------------------------------------------------------

    def batch_run(
        self,
        cypher: str,
        rows: list[dict[str, Any]],
        label: str,
        parameters: dict[str, Any] | None = None,
    ):
        if not rows:
            print(f"{label}: 0 rows，跳过")
            return

        total = len(rows)
        parameters = parameters or {}

        def _write(tx, current_batch):
            result = tx.run(
                cypher,
                rows=current_batch,
                kb_id=self.kb_id,
                run_id=self.run_id,
                **parameters,
            )
            return result.consume()

        with self._session() as session:
            for i in range(0, total, self.batch_size):
                batch = rows[i : i + self.batch_size]
                session.execute_write(_write, batch)
                print(
                    f"  {label}: "
                    f"{min(i + self.batch_size, total)} / {total}"
                )

    # ------------------------------------------------------------------
    # 导入行数据构造辅助方法
    # ------------------------------------------------------------------

    def _common_props(self, original_id: str) -> dict[str, Any]:
        return {
            "id": original_id,
            "kb_id": self.kb_id,
            "import_run_id": self.run_id,
            "updated_at": self.imported_at,
        }

    def _add_embedding(
        self,
        props: dict[str, Any],
        original_id: str,
    ) -> None:
        if self.import_vectors:
            vector = self.vector_map.get(str(original_id))
            if vector is not None:
                props["embedding"] = vector

    # ------------------------------------------------------------------
    # 1. 文档节点
    # ------------------------------------------------------------------

    def import_documents(self):
        print("\n--- 1. 导入 Documents ---")
        df = self.tables["documents"]
        rows = []

        for _, r in df.iterrows():
            oid = str(r["id"])
            props = self._common_props(oid)
            props.update(
                {
                    "title": _neo4j_value(r.get("title")),
                    "text": _neo4j_value(r.get("text")),
                    "human_readable_id": _neo4j_value(
                        r.get("human_readable_id")
                    ),
                    "creation_date": _neo4j_value(r.get("creation_date")),
                }
            )

            if "text_unit_ids" in df.columns:
                props["text_unit_ids"] = _id_list(r.get("text_unit_ids"))

            if "metadata" in df.columns:
                props["metadata"] = _neo4j_value(r.get("metadata"))

            rows.append(
                {
                    "uid": f"{self.kb_id}:{oid}",
                    "props": {k: v for k, v in props.items() if v is not None},
                }
            )

        cypher = """
        UNWIND $rows AS row
        MERGE (d:Document {uid: row.uid})
        SET d += row.props
        """
        self.batch_run(cypher, rows, "Documents")

    # ------------------------------------------------------------------
    # 2. 文本块节点与文档证据关系
    # ------------------------------------------------------------------

    def import_text_units(self):
        print("\n--- 2. 导入 TextUnits ---")
        df = self.tables["text_units"]
        rows = []

        for _, r in df.iterrows():
            oid = str(r["id"])
            props = self._common_props(oid)
            props.update(
                {
                    "human_readable_id": _neo4j_value(
                        r.get("human_readable_id")
                    ),
                    "text": _neo4j_value(r.get("text")),
                    "n_tokens": _neo4j_value(r.get("n_tokens")),
                }
            )
            self._add_embedding(props, oid)

            rows.append(
                {
                    "uid": f"{self.kb_id}:{oid}",
                    "props": {k: v for k, v in props.items() if v is not None},
                }
            )

        cypher = """
        UNWIND $rows AS row
        MERGE (t:TextUnit {uid: row.uid})
        SET t += row.props
        """
        self.batch_run(cypher, rows, "TextUnits")

        print("\n--- 2.1 建立 Document -> TextUnit 证据关系 ---")
        edge_rows = []
        for _, r in df.iterrows():
            tu_id = str(r["id"])
            for doc_id in _textunit_document_ids(r):
                edge_rows.append(
                    {
                        "document_uid": f"{self.kb_id}:{doc_id}",
                        "textunit_uid": f"{self.kb_id}:{tu_id}",
                    }
                )

        cypher_edges = """
        UNWIND $rows AS row
        MATCH (d:Document {uid: row.document_uid})
        MATCH (t:TextUnit {uid: row.textunit_uid})
        MERGE (d)-[r:HAS_PART]->(t)
        SET r.kb_id = $kb_id,
            r.import_run_id = $run_id
        """
        self.batch_run(cypher_edges, edge_rows, "HAS_PART")

    # ------------------------------------------------------------------
    # 3. 实体节点与文本块提及关系
    # ------------------------------------------------------------------

    def import_entities(self):
        print("\n--- 3. 导入 Entities ---")
        df = self.tables["entities"]
        rows = []

        for _, r in df.iterrows():
            oid = str(r["id"])
            props = self._common_props(oid)
            props.update(
                {
                    "name": _neo4j_value(r.get("title")),
                    "title": _neo4j_value(r.get("title")),
                    "type": _neo4j_value(r.get("type")),
                    "description": _neo4j_value(r.get("description")),
                    "human_readable_id": _neo4j_value(
                        r.get("human_readable_id")
                    ),
                    "frequency": _neo4j_value(r.get("frequency")),
                    "degree": _neo4j_value(r.get("degree")),
                    "x": _neo4j_value(r.get("x")),
                    "y": _neo4j_value(r.get("y")),
                }
            )
            self._add_embedding(props, oid)

            rows.append(
                {
                    "uid": f"{self.kb_id}:{oid}",
                    "props": {k: v for k, v in props.items() if v is not None},
                }
            )

        cypher = """
        UNWIND $rows AS row
        MERGE (e:Entity {uid: row.uid})
        SET e += row.props
        """
        self.batch_run(cypher, rows, "Entities")

        print("\n--- 3.1 建立 TextUnit -> Entity MENTIONS ---")
        edge_rows = []
        for _, r in df.iterrows():
            entity_id = str(r["id"])
            for tu_id in _id_list(r.get("text_unit_ids")):
                edge_rows.append(
                    {
                        "entity_uid": f"{self.kb_id}:{entity_id}",
                        "textunit_uid": f"{self.kb_id}:{tu_id}",
                    }
                )

        cypher_edges = """
        UNWIND $rows AS row
        MATCH (t:TextUnit {uid: row.textunit_uid})
        MATCH (e:Entity {uid: row.entity_uid})
        MERGE (t)-[r:MENTIONS]->(e)
        SET r.kb_id = $kb_id,
            r.import_run_id = $run_id
        """
        self.batch_run(cypher_edges, edge_rows, "MENTIONS")

    # ------------------------------------------------------------------
    # 4. 实体关系
    # ------------------------------------------------------------------

    def import_relationships(self):
        print("\n--- 4. 导入 Entity Relationships ---")
        df = self.tables["relationships"]
        rows = []

        for _, r in df.iterrows():
            oid = str(r["id"])
            props = self._common_props(oid)
            props.update(
                {
                    "description": _neo4j_value(r.get("description")),
                    "weight": _neo4j_value(r.get("weight")),
                    "combined_degree": _neo4j_value(
                        r.get("combined_degree")
                    ),
                    "human_readable_id": _neo4j_value(
                        r.get("human_readable_id")
                    ),
                    # 关系本身保留其来源文本块 ID，这是后续追溯原文证据的关键字段；
                    # 可据此定位具体支持该关系的源 TextUnit。
                    "text_unit_ids": _id_list(r.get("text_unit_ids")),
                }
            )

            rows.append(
                {
                    "uid": f"{self.kb_id}:{oid}",
                    "source": str(r["source"]),
                    "target": str(r["target"]),
                    "props": {k: v for k, v in props.items() if v is not None},
                }
            )

        # GraphRAG 最终关系表中的 source/target 是实体标题，而不是实体 UUID。
        # 使用 kb_id 将实体匹配范围限制在当前知识库，防止跨知识库串联。
        cypher = """
        UNWIND $rows AS row
        MATCH (source:Entity {kb_id: $kb_id, name: row.source})
        MATCH (target:Entity {kb_id: $kb_id, name: row.target})
        MERGE (source)-[r:RELATED {uid: row.uid}]->(target)
        SET r += row.props
        """
        self.batch_run(cypher, rows, "RELATED")

    # ------------------------------------------------------------------
    # 5. 社区与社区报告
    # ------------------------------------------------------------------

    def import_communities(self):
        if "communities" not in self.tables:
            print("\n--- 5. communities.parquet 不存在，跳过 Communities ---")
            return

        print("\n--- 5. 导入 Communities ---")
        df = self.tables["communities"]
        self._require_columns(
            df,
            "communities",
            ["id", "community", "entity_ids"],
        )

        rows = []
        for _, r in df.iterrows():
            oid = str(r["id"])
            community = str(_neo4j_value(r.get("community")))

            props = self._common_props(oid)
            props.update(
                {
                    "community": _neo4j_value(r.get("community")),
                    "human_readable_id": _neo4j_value(
                        r.get("human_readable_id")
                    ),
                    "level": _neo4j_value(r.get("level")),
                    "parent": _neo4j_value(r.get("parent")),
                    "children": _neo4j_value(r.get("children")),
                    "title": _neo4j_value(r.get("title")),
                    "period": _neo4j_value(r.get("period")),
                    "size": _neo4j_value(r.get("size")),
                }
            )

            rows.append(
                {
                    "community_key": f"{self.kb_id}:{community}",
                    "props": {k: v for k, v in props.items() if v is not None},
                }
            )

        cypher = """
        UNWIND $rows AS row
        MERGE (c:Community {community_key: row.community_key})
        SET c += row.props
        """
        self.batch_run(cypher, rows, "Communities")

        print("\n--- 5.1 建立 Entity -> Community ---")
        edge_rows = []
        for _, r in df.iterrows():
            community = str(_neo4j_value(r.get("community")))
            for entity_id in _id_list(r.get("entity_ids")):
                edge_rows.append(
                    {
                        "entity_uid": f"{self.kb_id}:{entity_id}",
                        "community_key": f"{self.kb_id}:{community}",
                    }
                )

        cypher_edges = """
        UNWIND $rows AS row
        MATCH (e:Entity {uid: row.entity_uid})
        MATCH (c:Community {community_key: row.community_key})
        MERGE (e)-[r:IN_COMMUNITY]->(c)
        SET r.kb_id = $kb_id,
            r.import_run_id = $run_id
        """
        self.batch_run(cypher_edges, edge_rows, "IN_COMMUNITY")

        self.import_community_reports()

    def import_community_reports(self):
        if "community_reports" not in self.tables:
            print("\n--- community_reports.parquet 不存在，跳过报告 ---")
            return

        print("\n--- 5.2 合并 Community Reports ---")
        df = self.tables["community_reports"]
        self._require_columns(
            df,
            "community_reports",
            ["id", "community", "title", "summary", "full_content"],
        )

        rows = []
        for _, r in df.iterrows():
            report_id = str(r["id"])
            community = str(_neo4j_value(r.get("community")))

            report_props = {
                "report_id": report_id,
                "report_human_readable_id": _neo4j_value(
                    r.get("human_readable_id")
                ),
                "report_title": _neo4j_value(r.get("title")),
                "summary": _neo4j_value(r.get("summary")),
                "full_content": _neo4j_value(r.get("full_content")),
                "rank": _neo4j_value(r.get("rank")),
                "rating_explanation": _neo4j_value(
                    r.get("rating_explanation")
                ),
                "findings": _neo4j_value(r.get("findings")),
                "full_content_json": _neo4j_value(
                    r.get("full_content_json")
                ),
                "report_period": _neo4j_value(r.get("period")),
                "report_size": _neo4j_value(r.get("size")),
                "import_run_id": self.run_id,
                "updated_at": self.imported_at,
            }

            self._add_embedding(report_props, report_id)

            rows.append(
                {
                    "community_key": f"{self.kb_id}:{community}",
                    "props": {
                        k: v
                        for k, v in report_props.items()
                        if v is not None
                    },
                }
            )

        # 为兼容旧版导入器的使用方式，继续将 Community 作为对外查询节点；
        # 社区报告字段以及报告向量直接挂载到对应 Community 节点。
        cypher = """
        UNWIND $rows AS row
        MATCH (c:Community {community_key: row.community_key})
        SET c += row.props
        """
        self.batch_run(cypher, rows, "CommunityReports")

    # ------------------------------------------------------------------
    # 可选的快照同步逻辑
    # ------------------------------------------------------------------

    def sync_stale_records(self):
        """
        删除属于当前知识库、但已经不再存在于本次 GraphRAG 快照中的旧数据。

        该操作不会清空整个 Neo4j 数据库，只处理带有当前 kb_id 的数据。
        在 Config 中设置 GRAPHRAG_SYNC_DELETE_STALE = True 后启用。
        """
        if not self.sync_delete_stale:
            return

        print("\n=== 同步删除当前知识库的过期数据 ===")
        with self._session() as session:
            # 先删除由本导入器管理、但本次快照中已经不存在的旧关系。
            result = session.run(
                """
                MATCH ()-[r]->()
                WHERE r.kb_id = $kb_id
                  AND coalesce(r.import_run_id, '') <> $run_id
                DELETE r
                RETURN count(r) AS deleted
                """,
                kb_id=self.kb_id,
                run_id=self.run_id,
            )
            print(
                "stale relationships:",
                result.single()["deleted"],
            )

            for label in ("Community", "Entity", "TextUnit", "Document"):
                result = session.run(
                    f"""
                    MATCH (n:{label})
                    WHERE n.kb_id = $kb_id
                      AND coalesce(n.import_run_id, '') <> $run_id
                    DETACH DELETE n
                    RETURN count(n) AS deleted
                    """,
                    kb_id=self.kb_id,
                    run_id=self.run_id,
                )
                print(f"stale {label:<12}: {result.single()['deleted']}")

    # ------------------------------------------------------------------
    # 导入结果统计与校验
    # ------------------------------------------------------------------

    def print_neo4j_stats(self):
        print("\n=== Neo4j 导入结果 ===")
        queries = {
            "Document": "MATCH (n:Document {kb_id:$kb}) RETURN count(n) AS c",
            "TextUnit": "MATCH (n:TextUnit {kb_id:$kb}) RETURN count(n) AS c",
            "Entity": "MATCH (n:Entity {kb_id:$kb}) RETURN count(n) AS c",
            "Community": "MATCH (n:Community {kb_id:$kb}) RETURN count(n) AS c",
            "HAS_PART": (
                "MATCH (:Document {kb_id:$kb})-[r:HAS_PART]->(:TextUnit) "
                "RETURN count(r) AS c"
            ),
            "MENTIONS": (
                "MATCH (:TextUnit {kb_id:$kb})-[r:MENTIONS]->(:Entity) "
                "RETURN count(r) AS c"
            ),
            "RELATED": (
                "MATCH (:Entity {kb_id:$kb})-[r:RELATED]->(:Entity) "
                "RETURN count(r) AS c"
            ),
            "IN_COMMUNITY": (
                "MATCH (:Entity {kb_id:$kb})-[r:IN_COMMUNITY]->(:Community) "
                "RETURN count(r) AS c"
            ),
        }

        with self._session() as session:
            for name, query in queries.items():
                count = session.run(query, kb=self.kb_id).single()["c"]
                print(f"{name:<18}: {count}")

    def clear_knowledge_base(self):
        """
        显式删除当前 GraphRAG 知识库的数据，不影响其他知识库。
        用于替代旧版危险的全库清空操作：MATCH (n) DETACH DELETE n。
        """
        print(f"🧹 删除知识库: {self.kb_id}")
        with self._session() as session:
            result = session.run(
                """
                MATCH (n)
                WHERE n.kb_id = $kb_id
                DETACH DELETE n
                RETURN count(n) AS deleted
                """,
                kb_id=self.kb_id,
            )
            print(f"删除节点: {result.single()['deleted']}")

    # ------------------------------------------------------------------
    # 完整导入流程
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        print("=" * 70)
        print("GraphRAG 3.1.2 -> Neo4j 增量导入")
        print("=" * 70)
        print(f"run_id     : {self.run_id}")
        print(f"kb_id      : {self.kb_id}")
        print(f"database   : {self.database}")
        print(f"vectors    : {self.import_vectors}")
        print(f"sync stale : {self.sync_delete_stale}")

        self.load_tables()
        self.validate()
        self.load_vectors()
        self.create_schema()

        self.import_documents()
        self.import_text_units()
        self.import_entities()
        self.import_relationships()
        self.import_communities()

        self.sync_stale_records()
        self.print_neo4j_stats()

        print("\n=== 导入完成 ===")
        return {
            "status": "success",
            "kb_id": self.kb_id,
            "run_id": self.run_id,
            "vectors": self.import_vectors,
            "vector_dim": self.vector_dim,
        }


# ----------------------------------------------------------------------
# LangChain 工具封装
# ----------------------------------------------------------------------

from langchain_core.tools import tool


@tool(
    "graphragToNeo4j",
    description=(
        "将 GraphRAG 3.1.2 输出增量/幂等导入 Neo4j，"
        "包含文档、文本块、实体、关系、社区和原文证据链。"
    ),
)
def graphRAG2Neo4j():
    """
    将当前 GraphRAG 输出增量写入 Neo4j。

    默认不会清空数据库。
    如需同步删除当前知识库中已经从最新 GraphRAG 快照消失的数据，
    在 Config 中设置 GRAPHRAG_SYNC_DELETE_STALE = True。
    """
    importer = GraphRAG2Neo4j(
        neo4j_uri=NEO4J_URI,
        neo4j_username=NEO4J_USERNAME,
        neo4j_password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE,
    )
    try:
        result = importer.run()
        return (
            "GraphRAG 数据已成功增量导入 Neo4j。"
            f" kb_id={result['kb_id']}, run_id={result['run_id']}"
        )
    finally:
        importer.close()


if __name__ == "__main__":
    importer = GraphRAG2Neo4j(
        neo4j_uri=NEO4J_URI,
        neo4j_username=NEO4J_USERNAME,
        neo4j_password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE,
    )

    try:
        importer.run()
    except Exception as e:
        print(f"\n❌ 导入失败: {type(e).__name__}: {e}")
        raise
    finally:
        importer.close()
