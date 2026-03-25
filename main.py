import os
from config import Config
from langchain_openai import ChatOpenAI

API_KEY = Config.LLM_API_KEY
BASE_URL = Config.LLM_BASE_URL
LLM_MODEL = Config.LLM_MODEL

# 创建 LLM
llm = ChatOpenAI(
    model = LLM_MODEL,  # 百炼平台上的模型名称，如 qwen-max, qwen-plus 等
    api_key = API_KEY,  
    base_url = BASE_URL, 
    temperature=0.7
)

# 调用模型
from langchain_core.messages import HumanMessage
response = llm.invoke([HumanMessage(content="你好，请介绍一下你自己。")])
print(response.content)