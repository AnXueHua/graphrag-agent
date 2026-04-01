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

google_llm = ChatGoogleGenerativeAI(
    credentials=credentials,
    model = Config.LLM_MODEL,  
    api_key = Config.LLM_API_KEY,  
    project = Config.PROJECT_ID,
    vertexai=True,
    temperature = 0,
)

# response = google_llm.invoke({"messages": [HumanMessage(content="你好，请介绍一下你自己。")]})
# print(response)