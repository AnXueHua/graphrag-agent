import os
from config import Config
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from tools.miner_api_parser import minerU_parser
from tools.graphrag2neo4j import graphRAG2Neo4j

### 创建 LLM
google_llm = ChatGoogleGenerativeAI(
    model = Config.LLM_MODEL,  
    api_key = Config.LLM_API_KEY,  
    project = Config.PROJECT_ID,
    temperature = 1,
)

agent = create_agent(
    model = google_llm,
    tools = [minerU_parser, graphRAG2Neo4j]
    )




