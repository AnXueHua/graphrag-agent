import os
from langchain_openai import ChatOpenAI

API_KEY = "sk-fc6e3b32697c4b5eb99e0b5d95adfa2e"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

llm = ChatOpenAI(
    model = "Qwen3-Max-2026-01-23",  # 百炼平台上的模型名称，如 qwen-max, qwen-plus 等
    api_key = API_KEY,  
    base_url = BASE_URL, 
    temperature=0.7
)

# 调用模型
from langchain_core.messages import HumanMessage
response = llm.invoke([HumanMessage(content="你好，请介绍一下你自己。")])
print(response.content)