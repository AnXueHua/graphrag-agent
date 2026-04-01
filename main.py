from agents.init_agent import agent

res = agent.invoke({"messages": [{"role": "user", "content": "解析原始文件夹中的文件。"}]})
print(res)