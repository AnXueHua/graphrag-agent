import os
from config import Config
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from google.oauth2 import service_account

credentials = service_account.Credentials.from_service_account_file(
    "C:/Users/15377/Downloads/google_key.json",
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
)

# 创建 LLM
llm = ChatOpenAI(
    model = LLM_MODEL,  # 百炼平台上的模型名称，如 qwen-max, qwen-plus 等
    api_key = API_KEY,  
    base_url = BASE_URL, 
    temperature=0.7
)

# 调用模型
response = agent.invoke({"messages": [HumanMessage(content="你好，请介绍一下你自己。")]})
print(response)