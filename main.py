from agents.init_agent import agent

res = agent.invoke({"messages":[{"role":"user","content":"你好"}]})
print(res)