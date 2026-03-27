import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件
# override=True 表示如果有同名系统环境变量，优先使用 .env 中的值
load_dotenv(override=True)

class Config:
    # 1. 基础路径配置
    BASE_DIR = Path(__file__).resolve().parent

    # 原始文件路径 (如果 .env 没配，就默认用 ./raw_files)
    RAW_FILES_DIR = os.getenv("RAW_FILES_DIR", str(BASE_DIR / "raw_files"))

    # GraphRAG 项目路径
    GRAPHRAG_ROOT = os.getenv("GRAPHRAG_ROOT", str(BASE_DIR / "graphrag"))
    GRAPHRAG_INPUT_DIR = os.path.join(GRAPHRAG_ROOT, "input")
    GRAPHRAG_OUTPUT_DIR = os.path.join(GRAPHRAG_ROOT, "output")
    GRAPHRAG_LANCEDB_DIR = os.path.join(GRAPHRAG_OUTPUT_DIR, "lancedb")

    # 2. MinerU 配置
    MINERU_API_KEY = os.getenv("MINERU_API_KEY")
    if not MINERU_API_KEY:
        raise ValueError("❌ 错误: 未在 .env 文件中找到 MINERU_API_KEY")

    # 3. Neo4j 配置
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USERNAME = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
    if not NEO4J_PASSWORD:
        raise ValueError("❌ 错误: 未在 .env 文件中找到 NEO4J_PASSWORD")

    # 4. LLM 配置
    LLM_API_KEY = os.getenv("LLM_API_KEY")
    if not LLM_API_KEY:
        raise ValueError("❌ 错误: 未在 .env 文件中找到 LLM_API_KEY")
    LLM_BASE_URL = os.getenv("LLM_BASE_URL")
    LLM_MODEL = os.getenv("LLM_MODEL")
    PROJECT_ID = os.getenv("PROJECT_ID")
    EMB_MODEL = os.getenv("EMB_MODEL")

# 确保必要的目录存在
os.makedirs(Config.RAW_FILES_DIR, exist_ok=True)
os.makedirs(Config.GRAPHRAG_INPUT_DIR, exist_ok=True)