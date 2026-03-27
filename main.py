import os
from config import Config
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

API_KEY = Config.LLM_API_KEY
PROJECT_ID = Config.PROJECT_ID
LLM_MODEL = Config.LLM_MODEL

# 创建 LLM
llm = ChatGoogleGenerativeAI(
    model = LLM_MODEL,  
    api_key = API_KEY,  
    project = PROJECT_ID,
    temperature = 1,
)

agent = create_agent(llm)

# 调用模型
response = agent.invoke({"messages": [HumanMessage(content="你好，请介绍一下你自己。")]})
print(response)