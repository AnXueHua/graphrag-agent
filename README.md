# graphrag-agent搭建文档

## 环境搭建

采用技术栈：dify + Microsoft GraphRAG + Neo4j

环境：windows 10 + docker desktop + jdk25 + python3.12

dify：https://github.com/langgenius/dify

docker：https://www.docker.com/products/docker-desktop/

Microsoft GraphRAG：https://github.com/microsoft/graphrag

Neo4j：https://neo4j.com/

mineru：https://mineru.net/

**使用前需要将所有.env.example复制为.env并填入自己的信息**

## 搭建知识图谱

### MINER

miner_api_parser.py：用于将pdf文件转为md文件类型，调用mineru api。（需要在.env文件进行配置）

### GraphRAG
```
python -m pip install graphrag
```
安装Microsoft GraphRAG后，在 graphrag 文件夹下进行初始化：

```
graphrag init --root ./graphrag/
```

修改settings.yaml文件内容（将openai模型换成其他可用模型，本项目示例为阿里千问）：

```
### LLM settings ###
models:
  default_chat_model:
    type: chat
    model_provider: openai  # 保持不变，利用兼容协议
    auth_type: api_key
    api_key: ${GRAPHRAG_API_KEY} # 在 .env 文件中填入你的阿里 API Key
    # 设置阿里的兼容接口地址
    api_base: ${GRAPHRAG_API_BASE}
    # 使用千问模型
    model: ${GRAPHRAG_CHAT_MODEL}
    model_supports_json: true 
    # 建议调低并发，以免触发阿里的限流 (默认25可能太高)
    concurrent_requests: 5 
    async_mode: threaded
    retry_strategy: exponential_backoff
    max_retries: 10
    
  default_embedding_model:
    type: embedding
    model_provider: openai # 保持不变
    auth_type: api_key
    api_key: ${GRAPHRAG_API_KEY}
    # 设置阿里的兼容接口地址
    api_base: ${GRAPHRAG_API_BASE}
    # 使用阿里的嵌入模型
    model: ${GRAPHRAG_EMBEDDING_MODEL}
    # 同样建议调低并发
    concurrent_requests: 5
    async_mode: threaded
    retry_strategy: exponential_backoff
    max_retries: 10
```

GraphRAG 会读取根目录下的 `.env` 文件。请确保你在这个文件中填入了阿里的 API Key：

1. 打开或创建 `.env` 文件。
2. 修改或添加以下行：

```
# 使用SDK调用时需配置的base_url
GRAPHRAG_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
GRAPHRAG_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxx  <-- 这里填你的阿里 DashScope API Key
# 对话模型名称
GRAPHRAG_CHAT_MODEL=qwen-plus-1220
# 嵌入模型名称
GRAPHRAG_EMBEDDING_MODEL=text-embedding-v1
```

注：使用阿里的模型时需要修改源码文件E:\miniconda3\envs\graphrag-agent\Lib\site-packages\graphrag\language_model\providers\litellm\embedding_model.py
找到函数_base_aembedding，修改代码：

```
new_args = {**self.kwargs, **kwargs}
new_args["encoding_format"] = "float"  # <--- 添加这一行，强制指定格式
return await aembedding(**new_args)
```

当转换后的文件如果为markdown格式时需要修改settings.yaml文件：

```
### Input settings ###

input:
  storage:
    type: file # or blob
    base_dir: "input"
  file_type: text # [csv, text, json]
  file_pattern: ".*(\\.txt|\\.md|\\.csv|\\.json)$$"
```

上述内容完成后运行以下代码测试：

```
graphrag index --root . --verbose
```

#### （可选）提示词优化

运行以下代码：

```
graphrag prompt-tune --root . --config ./settings.yaml --language Chinese --output ./prompts --discover-entity-types
```

修改settings.yaml中的\### Workflow settings ###部分，替换为生成的（覆盖后）的prompt文件，修改

```
extract_graph:
  model_id: default_chat_model
  prompt: "prompts/extract_graph.txt"
  # 修改entity_types内容，与extract_graph中的entity_types匹配
  entity_types: [algorithm, complexity_class, optimization_problem, mathematical_model, linear_programming, constraint, objective_function, solution_method, theoretical_concept, computational_problem]
  max_gleanings: 1
```

修改后的settings.yaml中### Workflow settings ###参考内容如下：

```
### Workflow settings ###
# 如果自动生成了prompt文件，请修改下面内容
embed_text:
  model_id: default_embedding_model
  vector_store_id: default_vector_store

extract_graph:
  model_id: default_chat_model
  prompt: "prompts/extract_graph.txt"
  entity_types: [algorithm, complexity_class, optimization_problem, mathematical_model, linear_programming, constraint, objective_function, solution_method, theoretical_concept, computational_problem]
  max_gleanings: 1

summarize_descriptions:
  model_id: default_chat_model
  prompt: "prompts/summarize_descriptions.txt"
  max_length: 500

extract_graph_nlp:
  text_analyzer:
    extractor_type: regex_english # [regex_english, syntactic_parser, cfg]
  async_mode: threaded # or asyncio

cluster_graph:
  max_cluster_size: 10

extract_claims:
  enabled: false
  model_id: default_chat_model
  prompt: "prompts/extract_claims.txt"
  description: "Any claims or facts that could be relevant to information discovery."
  max_gleanings: 1

community_reports:
  model_id: default_chat_model
  graph_prompt: "prompts/community_report_graph.txt"
  text_prompt: "prompts/community_report_graph.txt"
  max_length: 2000
  max_input_length: 8000

embed_graph:
  enabled: false # if true, will generate node2vec embeddings for nodes

umap:
  enabled: false # if true, will generate UMAP embeddings for nodes (embed_graph must also be enabled)

snapshots:
  graphml: true
  embeddings: true

### Query settings ###
## The prompt locations are required here, but each search method has a number of optional knobs that can be tuned.
## See the config docs: https://microsoft.github.io/graphrag/config/yaml/#query

local_search:
  chat_model_id: default_chat_model
  embedding_model_id: default_embedding_model
  prompt: "prompts/local_search_system_prompt.txt"

global_search:
  chat_model_id: default_chat_model
  map_prompt: "prompts/global_search_map_system_prompt.txt"
  reduce_prompt: "prompts/global_search_reduce_system_prompt.txt"
  knowledge_prompt: "prompts/global_search_knowledge_system_prompt.txt"

drift_search:
  chat_model_id: default_chat_model
  embedding_model_id: default_embedding_model
  prompt: "prompts/drift_search_system_prompt.txt"
  reduce_prompt: "prompts/drift_search_reduce_prompt.txt"

basic_search:
  chat_model_id: default_chat_model
  embedding_model_id: default_embedding_model
  prompt: "prompts/basic_search_system_prompt.txt"
```

### NEO4J
graphrag2neo4j.py：用于将 GraphRAG 生成的 Parquet 文件导入到 Neo4j 数据库中。

在使用前，请确保 `.env` 文件中包含以下 Neo4j 配置信息：

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
Neo4j_DATABASE = neo4j
```

运行以下命令将数据导入 Neo4j：

```bash
python graphrag2neo4j.py
```

该脚本会读取 `graphrag/output` 目录下的 Parquet 文件，并在 Neo4j 中创建以下节点和关系：
- **节点**: Entity, Community, TextUnit, Document
- **关系**: RELATED, IN_COMMUNITY, HAS_PART, MENTIONS

### searxng（联网搜索）

在 `docker-compose.yaml` 中添加 searxng 服务配置：

```yaml
searxng:
    image: searxng/searxng:latest
    container_name: searxng
    restart: always
    ports:
      - "8081:8080"
    volumes:
      - ./volumes/searxng-data:/etc/searxng
    environment:
      - SEARXNG_BASE_URL=http://0.0.0.0:8081/
    networks:
      - default
```

在挂载卷目录（如 `./volumes/searxng-data`）下新建 `settings.yaml` 文件，写入以下配置：

```yaml
use_default_settings: true
server:
  port: 8080
  bind_address: "0.0.0.0"
  secret_key: "replace_this_with_a_random_string_dify" # 随便写一串字符即可

search:
  formats:
    - html
    - json
```

重新启动docker

```bash
cd ~/dify/docker
docker-compose up -d
```
