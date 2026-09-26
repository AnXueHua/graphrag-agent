# GraphRAG Prompt 优化教程

> 本教程基于一个真实项目（电力设备技术规程知识图谱）的完整优化实践编写，适用于任何使用 Microsoft GraphRAG 构建领域知识图谱的开发者。
>
> 核心思想：**GraphRAG 的默认 prompt 是通用的，但你的数据是领域的。优化的本质是用 prompt 把领域约束"编译"进 LLM 的抽取行为中。**

---

## 0. 修改：示例生成的样本隔离（防"串样本"）

**文件**：`graphrag/prompt_tune/generator/entity_relationship.py` 的 `generate_entity_relationship_examples()`

### 修改原因

`prompt-tune` 为最终 `extract_graph.txt` 生成最多 5 个示例（`MAX_EXAMPLES = 5`），每个示例由一个独立的文档 chunk 生成。但 `CompletionMessagesBuilder` 是累积式的：`add_user_message()` 会把消息追加到同一个内部列表，`build()` 返回当前列表快照。

原版代码在列表推导式**外部**创建了一个共享的 `msg_builder`，导致第 N 个示例的请求携带了前 N-1 个 chunk 的 user 消息：

| 示例 | 实际发送给模型的消息 |
|------|---------------------|
| Example 1 | system(persona) + user(chunk 1) |
| Example 2 | system(persona) + user(chunk 1) + user(chunk 2) |
| Example 3 | system(persona) + user(chunk 1) + user(chunk 2) + user(chunk 3) |
| ... | ... |
| Example 5 | system(persona) + user(chunk 1..5) |

后果：生成 Example N 时模型能看到其他 chunk 的全文，输出中可能混入其他 chunk 的实体和关系，造成示例"串样本"——示例输出与示例标注的输入文本不对应。

### 修改方法

把 builder 的创建移进列表推导式，每个示例使用独立的 builder：

```python
# 修改前：共享 builder，user 消息跨迭代累积
msg_builder = CompletionMessagesBuilder().add_system_message(persona)

tasks = [
    model.completion_async(
        messages=msg_builder.add_user_message(message).build(),
        response_format_json_object=json_mode,
    )
    for message in messages
]

# 修改后：每个示例独立 builder，消息严格隔离
tasks = [
    model.completion_async(
        messages=CompletionMessagesBuilder()
            .add_system_message(persona)
            .add_user_message(message)
            .build(),
        response_format_json_object=json_mode,
    )
    for message in messages
]
```

修改后，第 N 个示例的请求消息严格为 `system(persona) + user(chunk N)`，每个示例只由它自己的 chunk 生成。

---

## 目录

0. [关键修复：示例生成的样本隔离（防"串样本"）](#0-关键修复示例生成的样本隔离防串样本)
1. [GraphRAG Prompt 体系全景](#1-graphrag-prompt-体系全景)
2. [优化方法论：十大技术](#2-优化方法论十大技术)
3. [实操步骤：从零到优化完成](#3-实操步骤从零到优化完成)
4. [完整示例：extract_graph.txt](#4-完整示例extract_graphtxt)
5. [完整示例：summarize_descriptions.txt](#5-完整示例summarize_descriptionstxt)
6. [完整示例：community_report_graph.txt](#6-完整示例community_report_graphtxt)
7. [元 Prompt 优化](#7-元-prompt-优化)
8. [验证与迭代](#8-验证与迭代)
9. [常见问题与陷阱](#9-常见问题与陷阱)
10. [检查清单](#10-检查清单)

---

## 1. GraphRAG Prompt 体系全景

### 1.1 索引阶段 Prompt

GraphRAG 索引流水线中，每个阶段都有独立的 prompt，由 `settings.yaml` 控制：

```
PDF/MD 文档
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  extract_graph          ← prompts/extract_graph.txt     │
│  (实体 + 关系抽取)        最核心、影响最大的 prompt       │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  summarize_descriptions   ← prompts/summarize_descriptions.txt │
│  (实体描述聚合)              将同一实体的多条描述合并     │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  community_reports        ← prompts/community_report_graph.txt  │
│  (社区报告)                  ← prompts/community_report_text.txt │
│                              对聚类后的社区生成摘要报告   │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  extract_claims (可选)    ← prompts/extract_claims.txt  │
│  (声明抽取)                    抽取文本中的声明/事实     │
└─────────────────────────────────────────────────────────┘
```

**优化优先级**：`extract_graph.txt` >> `summarize_descriptions.txt` > `community_report_graph.txt` > 其他

> `extract_graph.txt` 决定了知识图谱的"原材料"质量。如果实体抽取阶段就引入了噪声（幻觉实体、错误类型、虚假关系），后续所有阶段都无法修复。

### 1.2 查询阶段 Prompt

```yaml
# settings.yaml 中的查询配置
local_search:
  prompt: "prompts/local_search_system_prompt.txt"
global_search:
  map_prompt: "prompts/global_search_map_system_prompt.txt"
  reduce_prompt: "prompts/global_search_reduce_system_prompt.txt"
  knowledge_prompt: "prompts/global_search_knowledge_system_prompt.txt"
drift_search:
  prompt: "prompts/drift_search_system_prompt.txt"
basic_search:
  prompt: "prompts/basic_search_system_prompt.txt"
```

查询 prompt 通常不需要大幅修改，除非你的查询场景有特殊需求（如要求特定格式的回答、特定的引用方式）。

### 1.3 Prompt Tuning 元 Prompt

`graphrag prompt-tune` 命令内部使用一组**元 prompt**（meta-prompts）来自动生成上述 prompt。这些元 prompt 位于：

```
<python_env>/Lib/site-packages/graphrag/prompt_tune/prompt/
├── entity_relationship.py    # 生成实体/关系抽取示例的 prompt
├── entity_types.py           # 发现实体类型的 prompt
├── persona.py                # 生成领域专家角色的 prompt
├── domain.py                 # 识别文本领域的 prompt
├── language.py               # 检测文本语言的 prompt
├── community_reporter_role.py # 生成社区报告角色的 prompt
└── community_report_rating.py # 生成报告评分标准的 prompt
```

**关键认知**：`graphrag prompt-tune` 生成的 `extract_graph.txt` 中的示例（Examples）是由 `entity_relationship.py` 中的元 prompt 生成的。如果你只优化最终的 `extract_graph.txt` 而不优化元 prompt，下次重新运行 `prompt-tune` 时你的优化会被覆盖。

### 1.4 settings.yaml 中的 Prompt 配置

```yaml
extract_graph:
  completion_model_id: default_completion_model
  prompt: "prompts/frozen_v2/extract_graph.txt"  # 指向优化后的 prompt
  entity_types: ["设备", "设备部件", "组织机构", ...]
  max_gleanings: 1

summarize_descriptions:
  completion_model_id: default_completion_model
  prompt: "prompts/frozen_v2/summarize_descriptions.txt"
  max_length: 500

community_reports:
  completion_model_id: default_completion_model
  graph_prompt: "prompts/frozen_v2/community_report_graph.txt"
  text_prompt: "prompts/community_report_text.txt"
  max_length: 2000
  max_input_length: 8000
```

**最佳实践**：将优化后的 prompt 放在独立目录（如 `prompts/frozen_v2/`），通过 `settings.yaml` 的 `prompt` 字段指向它。这样：
- 保留官方 prompt 作为基线对照
- 支持多版本 A/B 对比
- 重新运行 `prompt-tune` 不会覆盖你的优化

---

## 2. 优化方法论：十大技术

以下十大技术从实际优化实践中提炼，按对抽取质量的影响程度排序。

### 技术 1：质量优先级排序（Quality Priority Ordering）

**问题**：默认 prompt 没有告诉模型"当规则冲突时，什么最重要"。模型可能为了多抽实体（高召回）而引入大量噪声。

**方法**：在 prompt 开头明确列出质量维度的优先级：

```text
抽取质量优先级：

1. 原文忠实性
2. 实体与关系准确性
3. 类型一致性
4. 实体粒度稳定性
5. 关系精度
6. 最后才是召回率

宁可遗漏弱实体或弱关系，也不得根据常识、上下文猜测或补充原文没有表达的知识。
```

**为什么有效**：
- 给模型一个"决策仲裁规则"——当"多抽一个实体"和"保持精确"冲突时，模型知道选后者
- "最后才是召回率"这句话直接改变了模型的抽取阈值
- 适用于任何领域，只需调整具体维度名称

**通用模板**：

```text
抽取质量优先级：

1. 原文忠实性（不添加原文没有的信息）
2. 实体与关系准确性（名称、类型、关系正确）
3. 类型一致性（同一概念在不同 chunk 中类型一致）
4. 实体粒度稳定性（实体名称跨文档可复用）
5. 关系精度（只抽取有明确文本支持的关系）
6. 最后才是召回率

宁可遗漏，不得臆造。
```

---

### 技术 2：源忠实性规则（Source Fidelity Rules / 反幻觉）

**问题**：LLM 会"好心"地补充常识性知识。例如文本只说"工作前应将弹簧释能"，模型会补充"弹簧是电磁操动机构的储能部件"。

**方法**：用专门的章节定义"负空间"——明确列出**不能**使用什么信息源：

```text
==================================================
2. Source Fidelity Rules
==================================================

所有实体、实体类型、实体描述和关系都只能依据当前 input_text。

不得使用：

- 其他文档的信息；
- 上一个或下一个 chunk 的内容；
- 文件名中隐含的信息；
- Examples 中出现但当前文本不存在的信息；
- 常识；
- 行业经验；
- 外部知识；
- 模型自身已有知识。

不得补充当前文本没有明确表达的：

- 全称；
- 简称；
- 别名；
- 上位概念；
- 下位概念；
- 设备用途；
- 工作原理；
- 重要性；
- 风险；
- 后果；
- 因果机制；
- 典型应用。

例如：

如果当前文本只写：
"工作前应将弹簧释能"

不能额外写：
"弹簧是电磁操动机构的储能部件"

除非当前文本明确表达该关系。

如果某项事实无法仅凭当前 input_text 支持，则不要输出。
```

**为什么有效**：
- LLM 对"不要做 X"的遵循度远高于"只做 Y"（当 Y 是复杂行为时）
- 列出具体禁止项比笼统说"不要幻觉"有效得多
- 给出 Bad/Good 对比示例让模型理解边界
- "文件名中隐含的信息"这一条特别重要——GraphRAG 的 chunk 可能包含文件名信息，模型会利用它

**通用化**：将禁止列表替换为你领域的典型幻觉模式。例如：
- 医疗领域：禁止补充"该药物通常用于治疗..."
- 法律领域：禁止补充"根据相关法律规定..."
- 学术领域：禁止补充"该方法的典型应用包括..."

---

### 技术 3：实体命名规则（Entity Name Rules）

**问题**：默认 prompt 只说"提取实体名称"，但没说什么**不能**作为实体名称。模型会抽取：
- 数值："15min"、"0.5%"、"220kV"
- 完整条件句："年漏气率小于0.5%"
- 泛化占位词："厂家"、"相关规程"、"工作人员"

**方法**：分四个子规则，每个都带 Bad/Good 示例：

```text
==================================================
3. Entity Name Rules
==================================================

entity_name 必须是稳定、可复用的业务概念名称。

优先使用当前文本中已经出现的规范名称。

保留原文中的：
- 中文名称；
- 英文缩写；
- 型号；
- 标准编号；
- 大小写形式。

不得擅自扩写或改写成原文不存在的正式名称。

--------------------------------------------------
3.1 禁止将数值和单位作为实体
--------------------------------------------------

以下内容不能单独成为 entity_name：
- 15min
- 30min
- 0.5%
- 25Hz
- 220kV
- ±2℃

数值、单位和范围应该保留在相应实体的 entity_description 中。

--------------------------------------------------
3.2 禁止将完整条件句或要求句作为实体
--------------------------------------------------

以下形式通常不能作为 entity_name：
- 年漏气率小于0.5%
- 受潮硅胶不超过2/3
- 气瓶底部不结霜
- 无锈蚀变形卡涩
- 符合相关技术要求

应该提取稳定概念。

Bad:
年漏气率小于0.5%

Good:
年漏气率

并在 description 中记录：
年漏气率应小于0.5%。

--------------------------------------------------
3.3 禁止泛化占位实体
--------------------------------------------------

通常不得抽取无法跨文档稳定识别的泛化占位词，例如：
- 厂家
- 相关厂家
- 相关单位
- 相关部门
- 相关规程
- 相关规定
- 相关标准
- 工作人员
- 专人
- 人员
- 设备
- 被测试设备
- 现场

只有当前文本给出具体、可识别名称，或者该词本身确实是稳定业务角色或概念时，才可以抽取。

--------------------------------------------------
3.4 不允许强制分类
--------------------------------------------------

如果某候选概念无法明确归入 [{entity_types}] 中的某一类型：
不要抽取该实体。

不得为了提高召回率，把它强行放入"最接近"的类型。
不得创造任何不在 [{entity_types}] 中的新类型。
```

**为什么有效**：
- 3.1 解决了"数值实体"问题——这是技术文档中最常见的噪声
- 3.2 的 Bad/Good 对比让模型理解"参数名"和"参数值"的分离
- 3.3 的禁止列表直接来自实际数据中的高频噪声
- 3.4 防止模型"为了分类而分类"，宁可漏抽

**通用化**：
- 3.1 的数值列表替换为你领域常见的数值模式
- 3.3 的占位词列表从你的实际数据中收集（跑一遍基线，统计高频噪声实体）

---

### 技术 4：类型边界规则（Type Boundary Rules）

**问题**：默认 prompt 只列出类型名称，不解释每个类型的边界。模型会：
- 把"现场检查"归入"检测方法"（应该是"检查方式"）
- 把"检测记录"归入"技术文档"（应该是"文件记录"）
- 把"紧急停运"归入"检修类别"（应该是"检修工艺"）

**方法**：为每个类型写一个边界定义，包含：
1. **定义**（一句话说明这个类型是什么）
2. **正面示例**（属于这个类型的典型实体）
3. **反面示例**（容易混淆但不属于这个类型的实体）
4. **交叉引用**（与相邻类型的区分规则）

```text
==================================================
4. Entity Type Boundary Rules
==================================================

以下规则用于稳定类型边界。

--------------------------------------------------
4.8 检测试验项目
--------------------------------------------------

回答"检测什么 / 试验什么"。

指可执行的检测、测量、试验或分析项目。

例如：
- 绝缘电阻测试
- 局部放电检测
- 油中溶解气体分析
- 红外测温

不要把仪器、参数或者检查方式归入此类型。

--------------------------------------------------
4.9 检测方法
--------------------------------------------------

回答"采用什么技术方法或原理进行检测"。

表示技术方法、检测原理或分析方法。

不要把：
- 现场检查
- 查阅资料
- PMS检查

归入 检测方法。

这些属于 检查方式。

--------------------------------------------------
4.10 检查方式
--------------------------------------------------

业务过程中采用的检查、核查或资料核验方式。

例如：
- 现场检查
- 查阅资料
- PMS检查

不要将"一般检测""精确检测"等参数适用条件随意归入检查方式。

--------------------------------------------------
4.12 技术参数
--------------------------------------------------

可被测量、限制、设定、观察或比较的稳定指标名称。

entity_name 只保留参数名称。

Good:
- 空间分辨率
- 温度分辨率
- 帧频
- 油温
- 年漏气率
- 绝缘电阻

Bad:
- 空间分辨率不大于1.5毫弧度
- 年漏气率小于0.5%
- 油温超过85℃
- 25Hz

参数值、单位、对象和适用条件必须放入 description。

--------------------------------------------------
4.13 判定标准
--------------------------------------------------

用于判断状态、结果、参数或项目是否符合要求的稳定判据或明确评价条件。

只有当当前文本明确将其作为可复用的判据、合格条件或评价规则时才抽取。

不得把所有"应""不得""完好""正常"等要求都变成 判定标准 实体。

通常以下内容不应单独建节点：
- 校验合格
- 外观完好
- 密封良好
- 动作可靠
- 安装牢固
- 无锈蚀
- 无破损

如果这些只是某设备的具体要求，应保留在设备、参数、工艺或关系 description 中。
```

**为什么有效**：
- "回答什么问题"的定义方式（如"回答'检测什么'"）比抽象定义更直观
- 反面示例直接来自实际数据中的混淆案例
- 交叉引用（"这些属于检查方式"）帮助模型建立类型间的对比
- 20 个类型 × 每个 5-15 行 = 约 200-300 行，在 token 预算内

**通用化**：
1. 先跑一遍基线 `prompt-tune`，收集实体类型
2. 人工审查抽取结果，记录每个类型的混淆案例
3. 为每个类型写定义 + 正面示例 + 反面示例
4. 特别关注**容易混淆的类型对**（如"检测方法"vs"检查方式"、"技术文档"vs"文件记录"）

**类型数量建议**：8-20 个。太少则粒度不够，太多则模型难以区分边界。

---

### 技术 5：关系抽取规则（Relationship Rules）

**问题**：默认 prompt 说"找出所有明确相关的实体对"，但模型会：
- 把同一表格中出现的两个实体连成关系（共现 ≠ 关系）
- 根据行业常识推断关系（"变压器一定安装在变电站里"）
- 在表格的 rowspan/colspan 结构中推断包含关系

**方法**：七条子规则：

```text
==================================================
7. Relationship Rules
==================================================

关系必须由当前 input_text 明确表达或直接支持。

--------------------------------------------------
7.1 Co-occurrence is NOT a relationship
--------------------------------------------------

以下情况本身不能证明两个实体存在关系：
- 同一句出现；
- 同一段出现；
- 同一列表出现；
- 同一表格行出现；
- 同一表格列出现；
- 同一表格栏目下出现；
- 同一章节出现；
- 同一类别出现；
- 使用相同检查要求；
- 使用相同判定条件。

不得仅根据共现生成关系。

--------------------------------------------------
7.2 Table hierarchy is NOT automatically a structural relationship
--------------------------------------------------

表格中的父级栏目、序号、分组标题或 rowspan/colspan 结构，
不能自动证明：
- 包含；
- 组成；
- 安装于；
- 属于；
- 所有权；
- 因果；
- 控制；
- 依赖。

只有当前文本明确表达上述语义时，才能生成相应关系。

--------------------------------------------------
7.3 No external relationship inference
--------------------------------------------------

不得根据行业常识推断：
- 某部件一定安装在某设备内；
- 某设备一定属于某系统；
- 某故障一定由某原因导致；
- 某工艺一定用于某设备；
- 某参数一定属于某设备；
- 两种设备通常一起使用。

如果当前文本没有支持，则不输出。

--------------------------------------------------
7.4 Relationship endpoints
--------------------------------------------------

source_entity 和 target_entity：
必须都已经作为 entity 输出。
名称必须与 entity_name 完全一致。
不得在 relationship 中创造新的实体名称。

--------------------------------------------------
7.5 Relationship description
--------------------------------------------------

relationship_description 应简洁说明：
当前文本具体表达了什么关系。
优先直接转述原文事实。

不要扩展为：
- 原文未说明的作用；
- 原文未说明的重要性；
- 推测的因果关系；
- 行业常识；
- 通用结论。

--------------------------------------------------
7.6 Relationship direction
--------------------------------------------------

在语义明确时：
优先将主体、适用对象、拥有者或执行对象作为 source_entity，
将参数、部件、文档、工艺、方法、目标对象等作为 target_entity。

--------------------------------------------------
7.7 Relationship strength
--------------------------------------------------

relationship_strength 必须是整数。

使用以下标准：
10: 当前文本明确、核心且无歧义地表达关系。
9:  当前文本明确直接表达关系。
8:  当前文本明确支持，关系较强。
7:  直接支持但属于次要关系。
6:  文本仍有明确证据，但关系较弱。
1-5: 不要输出该关系。

关系数量不是目标。
优先 Precision，而不是 Recall。
```

**为什么有效**：
- 7.1 的共现禁止列表直接针对表格密集型文档（技术规程、标准文档）
- 7.2 专门处理表格结构——这是技术文档中最常见的关系噪声来源
- 7.7 的"1-5 不要输出"直接砍掉了弱关系，大幅降低噪声
- "关系数量不是目标"这句话改变了模型的优化方向

**通用化**：
- 7.1 的共现场景列表根据你文档的结构调整（如果是纯文本，去掉表格相关项）
- 7.3 的外部推断列表替换为你领域的典型常识推断
- 7.7 的评分标准可以根据你的需求调整阈值（如"低于 7 不输出"）

---

### 技术 6：定量绑定（Quantitative Binding）

**问题**：技术文档中大量出现"参数 + 数值 + 单位 + 适用对象 + 适用条件"的组合。模型会：
- 把不同对象的数值混合（"便携式不大于1.5毫弧度，手持式不大于1.9毫弧度" → 合并成一个错误值）
- 丢失适用条件（"220kV 为 3 月" → 只保留"3 月"）

**方法**：

```text
--------------------------------------------------
5.1 Quantitative Binding
--------------------------------------------------

对于参数、阈值、周期、数量、时间、电压等级等定量要求：

必须保持以下信息的正确绑定：

参数 + 数值 + 单位 + 适用对象 + 适用条件

例如当前文本同时给出：

便携式红外热像仪空间分辨率不大于1.5毫弧度；
手持式红外热像仪空间分辨率不大于1.9毫弧度。

则：

entity_name:
空间分辨率

entity_description 必须分别保留：

便携式红外热像仪：不大于1.5毫弧度；
手持式红外热像仪：不大于1.9毫弧度。

不得把两个对象的数值混合。
```

**为什么有效**：
- 给出了具体的"绑定五元组"（参数+数值+单位+对象+条件）
- 用实际数据中的多对象参数作为示例
- "不得把两个对象的数值混合"是明确的禁止指令

**通用化**：适用于任何包含定量数据的领域（工程、医疗、金融、科学实验等）。

---

### 技术 7：输出协议（Output Protocol）

**问题**：模型会在输出中添加：
- Markdown 代码块（```）
- 标题（# 实体列表）
- 分析过程（"让我分析一下..."）
- 中文全角括号（（））
- 额外说明（"以下是抽取结果：")

这些都会导致下游解析失败。

**方法**：

```text
==================================================
8. Output Protocol
==================================================

先输出全部 entity，
再输出全部 relationship。

每条记录之间严格使用：
##
作为分隔符。

实体必须严格使用：
("entity"<|><entity_name><|><entity_type><|><entity_description>)

关系必须严格使用：
("relationship"<|><source_entity><|><target_entity><|><relationship_description><|><relationship_strength>)

必须使用英文半角：
(
)
"
<|>

不得使用中文全角括号或引号。

不得输出：
- Markdown 代码块；
- 标题；
- 分析过程；
- 解释；
- 编号；
- 前言；
- 后记；
- JSON；
- 额外说明。

最后一条记录之后输出：
<|COMPLETE|>

如果当前文本中没有符合要求的实体或关系，
只输出能够可靠抽取的内容。
不要为了满足数量而创造实体或关系。
```

**为什么有效**：
- 明确列出"不得输出"的格式元素
- 强调半角/全角字符——中文 LLM 特别容易输出全角括号
- "不要为了满足数量而创造实体或关系"是最后一道防线

---

### 技术 8：领域角色注入（Domain-Specific Persona）

**问题**：默认的 `summarize_descriptions.txt` 和 `community_report_graph.txt` 使用通用角色（"You are an expert in..."），没有领域上下文。模型在聚合描述和生成社区报告时缺乏领域视角。

**方法**：在 prompt 开头注入一个精确的领域角色描述：

```text
You are an expert in power grid equipment technical standards and knowledge 
architecture, specializing in the regulatory and procedural frameworks of 
China State Grid. You are skilled at mapping complex inter-document 
relationships, constructing domain ontologies, and analyzing the hierarchical 
and cross-referential structures within technical regulation systems that 
span equipment acceptance, operation and maintenance, testing, evaluation, 
repair, and trial procedures.
```

**角色描述的结构**（三句话模板）：

```
You are an expert in [领域]. 
You are skilled at [核心技能1], [核心技能2], and [核心技能3]. 
You are adept at helping people [具体任务].
```

**为什么有效**：
- 角色描述影响模型的"注意力分布"——它会优先关注与角色相关的信息
- 对于 `summarize_descriptions`，领域角色帮助模型在合并描述时保留领域关键信息
- 对于 `community_report`，领域角色影响报告的视角和深度

**通用化**：
- 电力领域：`expert in power grid equipment technical standards...`
- 医疗领域：`expert in clinical trial protocols and medical device regulations...`
- 法律领域：`expert in corporate compliance and regulatory frameworks...`
- 学术领域：`expert in [具体学科] research methodology and literature analysis...`

---

### 技术 9：评分标准定制（Customized Rating Criteria）

**问题**：默认的 `community_report_graph.txt` 中的 REPORT RATING 是通用的"impact severity"，与你的领域无关。模型不知道什么在你的领域中是"重要的"。

**方法**：将评分标准替换为领域特定的多维度评估：

```text
- REPORT RATING: A float score between 0-10 that represents the relevance 
  of the text to China State Grid substation equipment technical regulation 
  systems, including:
  - the completeness and depth of coverage across acceptance, operation and 
    maintenance, testing, evaluation, repair, and trial procedures;
  - the clarity of hierarchical document structures and cross-referential 
    relationships between volumes (分册);
  - the specificity of technical specifications, test methodologies, 
    acceptance criteria, and scoring frameworks;
  - the utility for constructing domain ontologies and mapping inter-document 
    dependency chains within the "五通一措" regulatory framework,
  with 1 being trivial or irrelevant to the technical regulation system 
  and 10 being highly significant, comprehensive, and directly foundational 
  to understanding the complete lifecycle management architecture of 
  substation equipment.
```

**结构**：

```
A float score between 0-10 that represents the relevance of the text to 
[领域], including:
- [维度1：覆盖完整性]
- [维度2：结构清晰度]
- [维度3：技术具体性]
- [维度4：对下游任务的效用]
with 1 being [最低分定义] and 10 being [最高分定义].
```

**为什么有效**：
- 评分标准影响社区报告的"筛选"——低分社区会被降权
- 多维度评估比单一"重要性"更稳定
- 明确定义 1 分和 10 分让模型的评分更一致

---

### 技术 10：元 Prompt 优化（Meta-Prompt Optimization）

**问题**：`graphrag prompt-tune` 生成的示例质量取决于元 prompt。默认元 prompt 太简单，生成的示例：
- 不包含领域约束
- 没有 Bad/Good 对比
- 没有反幻觉规则
- 示例文本是玩具级别（"Alice sent an email to Bob"）

**方法**：重写 `site-packages/graphrag/prompt_tune/prompt/` 中的元 prompt。

详见 [第 7 节：元 Prompt 优化](#7-元-prompt-优化)。

---

## 3. 实操步骤：从零到优化完成

### Step 1：运行官方 prompt-tune 获取基线

```bash
cd graphrag

# 首次初始化
graphrag init --root ./

# 运行 prompt-tune（使用你的领域文档）
graphrag prompt-tune --root . --config ./settings.yaml --language Chinese --output ./prompts --discover-entity-types
```

这会生成 3 个索引阶段 prompt：
- `prompts/extract_graph.txt`（含自动生成的示例）
- `prompts/summarize_descriptions.txt`
- `prompts/community_report_graph.txt`

> 注意：查询阶段 prompt（`local_search_system_prompt.txt` 等）由 `graphrag init` 生成，不在 prompt-tune 的输出范围内。`community_report_text.txt` 也是 init 生成的（text 模式社区报告）。

**记录基线**：将生成的 prompt 复制到 `prompts/frozen_v1/` 作为对照。

### Step 2：小规模试跑，收集问题

```bash
# 用少量文档（5-10 个）试跑索引
graphrag index --root .
```

检查 `output/` 中的 Parquet 文件，重点关注：
- **实体噪声**：数值实体、句子实体、泛化占位词
- **类型混淆**：同一概念在不同 chunk 中被赋予不同类型
- **关系噪声**：共现关系、常识推断关系
- **描述幻觉**：添加了原文没有的信息

**收集问题清单**：按类型分类，每个问题记录：
- 问题描述
- 出现频率（高/中/低）
- 典型示例（原文 + 错误输出）

### Step 3：编写优化后的 extract_graph.txt

这是最核心的一步。按照以下结构组织：

```
1. Goal（目标 + 质量优先级）
2. Entity Extraction（实体抽取格式）
3. Source Fidelity Rules（源忠实性规则）
4. Entity Name Rules（实体命名规则）
5. Entity Type Boundary Rules（类型边界规则）
6. Description Rules（描述规则 + 定量绑定）
7. Relationship Extraction（关系抽取格式）
8. Relationship Rules（关系规则）
9. Output Protocol（输出协议）
10. Examples（2-3 个真实领域示例）
11. Real Data（实际数据占位符）
```

**关键原则**：
- 每个规则都要有**具体示例**（Bad/Good 对比）
- 示例必须来自**你的实际数据**，不要用玩具文本
- 规则之间不要矛盾
- 总长度控制在 3000-5000 tokens（加上示例约 5000-8000 tokens）

### Step 4：优化 summarize_descriptions.txt

在官方模板前注入领域角色：

```text
[领域角色描述]
Using your expertise, you're asked to generate a comprehensive summary...
[官方模板其余部分保持不变]
```

### Step 5：优化 community_report_graph.txt

1. 注入领域角色
2. 替换 REPORT RATING 标准为领域特定版本
3. 保持 JSON 输出格式不变

### Step 6：优化元 Prompt（可选但推荐）

如果你计划多次运行 `prompt-tune`，优化元 prompt 可以确保生成的示例质量。详见第 7 节。

### Step 7：验证与迭代

```bash
# 1. 用优化后的 prompt 重新索引
graphrag index --root .

# 2. 对比新旧输出
#    - 实体数量变化
#    - 类型分布变化
#    - 关系数量变化
#    - 抽样检查实体/关系质量

# 3. 如果发现问题，调整 prompt 并重复
```

**版本管理**：
```
prompts/
├── extract_graph.txt          # 当前使用的（指向 frozen_v2）
├── frozen_v1/                 # 第一次优化
│   ├── extract_graph.txt
│   ├── summarize_descriptions.txt
│   └── community_report_graph.txt
├── frozen_v2/                 # 第二次优化（当前）
│   ├── extract_graph.txt
│   ├── summarize_descriptions.txt
│   └── community_report_graph.txt
└── ...
```

---

## 4. 完整示例：extract_graph.txt

以下是一个完整的优化后 `extract_graph.txt` 的结构模板（以电力设备领域为例，其他领域替换具体内容即可）：

```text
-Goal-

给定一段当前文本 input_text，以及允许使用的实体类型 entity_types，
从当前文本中识别具有稳定业务语义的实体，并抽取这些实体之间由当前文本直接支持的关系。

本任务用于构建[领域]知识图谱。

抽取质量优先级：

1. 原文忠实性
2. 实体与关系准确性
3. 类型一致性
4. 实体粒度稳定性
5. 关系精度
6. 最后才是召回率

宁可遗漏弱实体或弱关系，也不得根据常识、上下文猜测或补充原文没有表达的知识。


==================================================
1. Entity Extraction
==================================================

识别当前 input_text 中属于以下允许类型的实体：

[{entity_types}]

对于每个实体输出：

- entity_name:
  稳定、可复用、由当前文本直接支持的实体名称。

- entity_type:
  必须严格等于 [{entity_types}] 中的某一个类型名称。

- entity_description:
  对当前文本中与该实体直接相关事实的简洁汇总。

格式：

("entity"<|><entity_name><|><entity_type><|><entity_description>)


==================================================
2. Source Fidelity Rules
==================================================

[技术 2 的内容]


==================================================
3. Entity Name Rules
==================================================

[技术 3 的内容]


==================================================
4. Entity Type Boundary Rules
==================================================

[技术 4 的内容，每个类型一个子节]


==================================================
5. Description Rules
==================================================

entity_description 应：
- 简洁；
- 事实化；
- 可追溯到当前文本；
- 优先保留有业务价值的信息；
- 保留重要定量条件。

不要写宣传性、解释性或总结性语言。

禁止自动加入类似：
- 是关键设备
- 是核心环节
- 是重要组成部分
- 对安全运行具有重要意义
- 保障设备安全稳定运行
- 确保系统可靠性

除非这些表述在当前文本中明确出现。

[技术 6：定量绑定]


==================================================
6. Relationship Extraction
==================================================

只在已经识别的实体之间抽取关系。

每个关系包含：
- source_entity
- target_entity
- relationship_description
- relationship_strength

格式：
("relationship"<|><source_entity><|><target_entity><|><relationship_description><|><relationship_strength>)


==================================================
7. Relationship Rules
==================================================

[技术 5 的内容]


==================================================
8. Output Protocol
==================================================

[技术 7 的内容]


==================================================
Examples
==================================================

Example 1:

entity_types: [{entity_types}]

text:
[你的实际领域文本片段 1]

output:
[正确的抽取结果 1]


Example 2:

entity_types: [{entity_types}]

text:
[你的实际领域文本片段 2]

output:
[正确的抽取结果 2]


==================================================
Real Data
==================================================

entity_types: [{entity_types}]

text:
{input_text}

output:
```

**示例编写要点**：

1. **使用真实数据**：从你的 `input/` 目录中选取有代表性的文本片段
2. **覆盖典型场景**：
   - 示例 1：包含参数 + 数值 + 多对象绑定（展示定量绑定）
   - 示例 2：包含工艺/操作 + 安全要求（展示关系方向）
3. **展示"不抽取"**：示例中应该有一些文本内容**没有**被抽取（展示源忠实性）
4. **关系强度分布**：示例中应包含 10、9、8 分的关系，但不包含 6 分以下的
5. **类型覆盖**：示例应覆盖 5-8 个不同类型

---

## 5. 完整示例：summarize_descriptions.txt

```text
You are an expert in [领域描述]. You are skilled at [技能1], [技能2], and [技能3]. You are adept at helping people [具体任务].
Using your expertise, you're asked to generate a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, concise description in Chinese. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we have the full context.

Enrich it as much as you can with relevant information from the nearby text, this is very important.

If no answer is possible, or the description is empty, only convey information that is provided within the text.
#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
```

**修改点**：
- 第一行：注入领域角色（替换默认的通用角色）
- 其余部分：保持官方模板不变

**角色描述示例**：

| 领域 | 角色描述 |
|------|---------|
| 电力设备 | `You are an expert in power grid equipment technical standards and knowledge architecture, specializing in the regulatory and procedural frameworks of China State Grid.` |
| 医疗 | `You are an expert in clinical medicine and medical device regulations, specializing in diagnostic protocols, treatment guidelines, and adverse event reporting.` |
| 法律 | `You are an expert in corporate law and regulatory compliance, specializing in contract analysis, liability frameworks, and regulatory interpretation.` |
| 学术 | `You are an expert in [学科] research, specializing in methodology analysis, experimental design, and literature synthesis.` |

---

## 6. 完整示例：community_report_graph.txt

在官方模板基础上修改两处：

### 修改 1：注入领域角色（第一行）

```text
You are an expert in [领域描述]. You are skilled at [技能1], [技能2], and [技能3]. You are adept at helping people [具体任务].
```

### 修改 2：替换 REPORT RATING 标准

**官方默认**：
```text
- REPORT RATING: A float score between 0-10 that represents the impact 
  severity of the community, with 1 being trivial and 10 being highly 
  significant.
```

**优化后**（以电力设备为例）：
```text
- REPORT RATING: A float score between 0-10 that represents the relevance 
  of the text to [领域], including:
  - [维度1：覆盖完整性]
  - [维度2：结构清晰度]
  - [维度3：技术具体性]
  - [维度4：对下游任务的效用],
  with 1 being trivial or irrelevant to [领域] 
  and 10 being highly significant, comprehensive, and directly foundational 
  to [领域的核心目标].
```

**其余部分保持官方模板不变**（JSON 格式、Grounding Rules、Example Input 等）。

---

## 7. 元 Prompt 优化

### 7.1 为什么需要优化元 Prompt

`graphrag prompt-tune` 的实际工作流程（源码：`graphrag/api/prompt_tune.py` 的 `generate_indexing_prompts`）：

```
0. 分块加载文档（默认取 15 个 chunk）
1. 识别领域              ← prompt/domain.py
2. 检测语言              ← prompt/language.py
3. 生成专家角色 persona   ← prompt/persona.py
4. 生成社区报告评分标准    ← prompt/community_report_rating.py
5. 发现实体类型           ← prompt/entity_types.py
6. 生成抽取示例           ← prompt/entity_relationship.py
7. 组装 extract_graph.txt ← template/extract_graph.py（把示例嵌入模板）
8. 组装 summarize_descriptions.txt ← template/entity_summarization.py（注入 persona）
9. 生成社区报告角色        ← prompt/community_reporter_role.py
10. 组装 community_report_graph.txt ← template/community_report_summarization.py
    （注入 persona + 报告角色 + 第 4 步的评分标准）
```

**关键依赖关系**：

- 第 6 步生成的示例会被嵌入到最终的 `extract_graph.txt` 中。如果元 prompt 太简单，生成的示例：
  - 不包含领域约束
  - 没有反幻觉规则
  - 示例文本是玩具级别
  - 没有 Bad/Good 对比
- 第 3 步的 persona 会被注入到 `summarize_descriptions.txt` 和 `community_report_graph.txt` 的开头
- 第 4 步的评分标准会被注入到 `community_report_graph.txt` 的 REPORT RATING 部分
- 第 5 步的实体类型会作为 `[{entity_types}]` 填入 `extract_graph.txt`

**结论**：最终 prompt 的每个"领域化"部分都来自某个元 prompt。只改最终 prompt 是"治标"，改元 prompt 才是"治本"（重新运行 prompt-tune 时优化不会丢失）。

### 7.2 优化 entity_relationship.py

**文件位置**：`<python_env>/Lib/site-packages/graphrag/prompt_tune/prompt/entity_relationship.py`

**官方默认**（简化版）：
```python
ENTITY_RELATIONSHIPS_GENERATION_PROMPT = """
-Goal-
Given a text document that is potentially relevant to this activity and a list
of entity types, identify all entities of those types from the text and all
relationships among the identified entities.

-Steps-
1. Identify all entities...
2. Identify all relationships...
3. Return output in {language}...
"""
```

**优化后**（核心结构）：

```python
ENTITY_RELATIONSHIPS_GENERATION_PROMPT = """
-Goal-

Given a text document that is potentially relevant to this activity and a list
of entity types, identify entities of those types from the CURRENT INPUT TEXT
and identify meaningful relationships among those entities.

The extraction should prioritize:
- source fidelity;
- stable entity granularity;
- semantic consistency;
- relationship quality;
- cross-document reusability.

-Entity Extraction Rules-

1. CURRENT INPUT TEXT ONLY
   [源忠实性规则]

2. EXTRACT STABLE DOMAIN OBJECTS
   [稳定实体定义]

3. ENTITY NAMES MUST REPRESENT BUSINESS CONCEPTS, NOT COMPLETE SENTENCES
   [命名规则 + Bad/Good 示例]

4. PARAMETER GRANULARITY
   [参数粒度规则]

5. VALUES ARE NOT STANDALONE ENTITIES
   [数值禁止规则]

6. CRITERIA AND THRESHOLDS
   [判定标准规则]

7. DO NOT NODE GENERIC QUALIFICATION PHRASES
   [泛化短语禁止规则]

8. PRESERVE ENTITY-TYPE BOUNDARIES
   [类型边界规则]

9. USE SOURCE-SUPPORTED NAMES
   [名称忠实性规则]

10. DESCRIPTION FIDELITY
    [描述忠实性规则]

11. QUANTITATIVE CONDITION BINDING
    [定量绑定规则]

12. AVOID DUPLICATES
    [去重规则]

-Relationship Quality Rules-

1. CURRENT INPUT TEXT ONLY
2. BOTH ENDPOINTS MUST EXIST
3. CO-OCCURRENCE IS NOT A RELATIONSHIP
4. DO NOT INFER DOMAIN RELATIONSHIPS
5. PREFER EXPLICIT DOMAIN SEMANTICS
6. AVOID REDUNDANT RELATIONSHIPS
7. DO NOT OMIT CLEAR RELATIONSHIPS
8. RELATIONSHIP DESCRIPTIONS MUST BE TRACEABLE
9. RELATIONSHIP STRENGTH
   [评分标准：9-10 核心关系，7-8 直接关系，6 次要关系，1-5 不输出]
10. QUALITY OVER QUANTITY

-Steps-
1. Identify all entities...
2. Identify meaningful relationships...
3. Return the output in {language}...
4. Final validation before output
   [输出前自检清单]
5. When finished, output <|COMPLETE|>

-Examples-
[3 个领域示例，每个包含：实体类型列表、文本、正确输出]
"""
```

**关键设计**：
- 12 条实体规则 + 10 条关系规则，与最终 `extract_graph.txt` 的规则**保持一致**
- 3 个示例使用**领域文本**（不是 "Alice sent an email to Bob"）
- 示例中的实体类型使用**中文**（与最终 prompt 一致）
- 包含"Final validation"步骤，让模型在输出前自检

### 7.3 优化 entity_types.py

**文件位置**：`<python_env>/Lib/site-packages/graphrag/prompt_tune/prompt/entity_types.py`

**官方默认**（简化版）：
```python
ENTITY_TYPE_GENERATION_PROMPT = """
The goal is to identify a stable, reusable, and semantically meaningful set of
entity types from the provided text for downstream entity and relationship extraction.
...
"""
```

**优化后**（16 条要求）：

```python
ENTITY_TYPE_GENERATION_PROMPT = """
The goal is to identify a stable, reusable, and semantically meaningful set of
entity types from the provided text for downstream entity and relationship extraction.
The user's task is:
{task}
You should infer the domain from the task and the REAL DATA itself.

IMPORTANT REQUIREMENTS FOR ENTITY TYPES:
1. Language
   All entity type names MUST be written in [目标语言].
2. Entity types must be categories, not entities
   Good: "组织机构" "设备" "技术参数"
   Bad:  "国家电网公司" "某型号变压器" "温度为80℃"
3. Prefer stable and reusable business/domain concepts
4. Prefer medium-grained entity types
   Avoid: "实体" "对象" "其他" (太宽)
   Avoid: 为每个子类型创建不同类型 (太细)
5. Do not use complete sentences or rules as entity types
6. Do not use parameter values as entity types
7. Do not confuse entity types with relationships
8. Do not confuse entity types with generic actions
9. Avoid redundant or semantically overlapping entity types
10. Preserve meaningful domain distinctions
11. Prefer semantic role over wording
12. Source fidelity
13. Cross-document consistency
14. Candidate semantic dimensions
    [列出可能的语义维度，但强调"不是强制列表"]
15. Quantity
    8-20 个高质量类型通常优于大量碎片化类型
16. Final check
    [输出前自检清单]

EXAMPLE SECTION:
[3 个示例，展示期望的抽象级别]

REAL DATA:
Task: {task}
Text: {input_text}
RESPONSE: {{<entity_types>}}
"""
```

**关键设计**：
- 第 1 条强制使用目标语言（中文）
- 第 2 条用 Good/Bad 对比区分"类型"和"实体"
- 第 9 条和第 10 条平衡"合并"和"区分"
- 第 14 条列出候选维度但强调"不是强制列表"——避免模型照搬
- 第 15 条给出数量建议（8-20）

### 7.4 注意事项

**修改 site-packages 的风险**：
- 升级 GraphRAG 版本时，修改会被覆盖
- 建议将修改后的文件备份到项目目录中

**备份策略**：
```bash
# 将修改后的元 prompt 备份到项目目录
cp <python_env>/Lib/site-packages/graphrag/prompt_tune/prompt/entity_relationship.py \
   graphrag/prompt_tune_backup/entity_relationship.py
cp <python_env>/Lib/site-packages/graphrag/prompt_tune/prompt/entity_types.py \
   graphrag/prompt_tune_backup/entity_types.py
```

**替代方案**：如果不想修改 site-packages，可以：
1. 运行 `prompt-tune` 生成基线
2. 手动编写 `extract_graph.txt`（不依赖自动生成的示例）
3. 在 `settings.yaml` 中直接指向手动编写的 prompt

---

## 8. 验证与迭代

### 8.1 自动化检查

```python
import pandas as pd

# 读取实体
entities = pd.read_parquet("output/entities.parquet")
relationships = pd.read_parquet("output/relationships.parquet")

# 1. 检查数值实体
numeric_pattern = r'^[\d\.\s%℃°kVHzmin±/]+$'
numeric_entities = entities[entities['title'].str.match(numeric_pattern, na=False)]
print(f"数值实体: {len(numeric_entities)}")
if not numeric_entities.empty:
    print(numeric_entities['title'].tolist()[:20])

# 2. 检查句子实体（长度 > 15 字符且包含"应"/"不得"/"小于"/"大于"）
sentence_pattern = r'(应|不得|小于|大于|不超过|不低于)'
sentence_entities = entities[
    (entities['title'].str.len() > 15) & 
    (entities['title'].str.contains(sentence_pattern, na=False))
]
print(f"疑似句子实体: {len(sentence_entities)}")

# 3. 检查类型分布
print("\n类型分布:")
print(entities['title'].value_counts())  # 注意：这里应该是 type 列

# 4. 检查关系强度分布
print("\n关系强度分布:")
print(relationships['strength'].value_counts().sort_index())

# 5. 检查弱关系（strength < 6）
weak_rels = relationships[relationships['strength'] < 6]
print(f"\n弱关系 (strength < 6): {len(weak_rels)}")
```

### 8.2 人工抽样检查

从每个类型中随机抽取 5-10 个实体，检查：
- [ ] 实体名称是否是稳定的业务概念（不是数值、句子、占位词）
- [ ] 类型是否正确（对照类型边界规则）
- [ ] 描述是否忠实于原文（没有添加外部知识）
- [ ] 定量信息是否正确绑定（参数+数值+单位+对象+条件）

从关系中随机抽取 10-20 条，检查：
- [ ] 两个端点是否都是已抽取的实体
- [ ] 关系是否有明确的文本支持（不是共现、不是常识推断）
- [ ] 关系方向是否合理
- [ ] 关系强度是否合理

### 8.3 A/B 对比

```bash
# 使用 frozen_v1 的 prompt
# settings.yaml: prompt: "prompts/frozen_v1/extract_graph.txt"
graphrag index --root .
# 保存输出到 output_v1/

# 使用 frozen_v2 的 prompt
# settings.yaml: prompt: "prompts/frozen_v2/extract_graph.txt"
graphrag index --root .
# 保存输出到 output_v2/

# 对比
python compare_outputs.py output_v1/ output_v2/
```

**对比指标**：
| 指标 | 说明 |
|------|------|
| 实体总数 | 变化幅度（v2 应该比 v1 少，因为去除了噪声） |
| 数值实体数 | 应该接近 0 |
| 句子实体数 | 应该接近 0 |
| 类型一致性 | 同一概念在不同 chunk 中的类型是否一致 |
| 关系总数 | 变化幅度（v2 应该比 v1 少，因为去除了弱关系） |
| 弱关系数 (strength < 6) | 应该接近 0 |
| 人工评分 | 抽样 50 个实体 + 50 条关系，评分 1-5 |

### 8.4 迭代策略

```
第 1 轮：解决最严重的噪声（数值实体、句子实体、共现关系）
    ↓
第 2 轮：解决类型混淆（调整类型边界规则）
    ↓
第 3 轮：优化描述质量（定量绑定、描述忠实性）
    ↓
第 4 轮：微调关系强度阈值
    ↓
稳定后：冻结为 frozen_vN，不再修改
```

**每轮迭代**：
1. 修改 prompt
2. 用同一批文档重新索引
3. 运行自动化检查
4. 人工抽样
5. 记录变化
6. 如果改善，冻结新版本；如果退步，回滚

---

## 9. 常见问题与陷阱

### Q1：prompt 太长，模型"忘记"前面的规则

**现象**：模型遵循了后面的规则但忽略了前面的规则。

**解决**：
- 将最重要的规则放在 prompt 的**开头**（质量优先级、源忠实性）
- 在 prompt 的**结尾**（Real Data 之前）重复关键约束
- 控制总长度在 8000 tokens 以内（包括示例）
- 如果类型很多（>15 个），考虑减少每个类型的描述长度

### Q2：模型仍然输出中文全角括号

**解决**：
- 在 Output Protocol 中明确列出半角字符：`( ) " <|>`
- 在示例中使用半角字符（模型会模仿示例的格式）
- 在下游解析代码中添加全角→半角的转换作为兜底

### Q3：模型在描述中添加"这是关键设备"等宣传性语言

**解决**：
- 在 Description Rules 中明确列出禁止的宣传性短语
- 在示例中展示"朴素"的描述风格（只陈述事实，不评价重要性）

### Q4：同一实体在不同 chunk 中被赋予不同类型

**解决**：
- 这是 chunk 级别的限制——每个 chunk 独立抽取，模型看不到其他 chunk
- 在类型边界规则中加强易混淆类型的区分
- 在 `summarize_descriptions` 阶段，模型会合并同一实体的描述，可以部分缓解
- 在下游 ETL 中，对同一名称的实体做类型投票（取多数）

### Q5：模型抽取了 Examples 中的实体

**解决**：
- 在 Source Fidelity Rules 中明确写"不得使用 Examples 中出现但当前文本不存在的信息"
- 示例使用与 Real Data 完全不同的文本
- 示例中的实体名称不要与 Real Data 中的实体名称重叠

### Q6：prompt-tune 生成的示例质量差

**解决**：
- 优化元 prompt（第 7 节）
- 或者：手动编写示例，不依赖 prompt-tune 的自动生成
- 手动编写示例时，从实际数据中选取 2-3 个有代表性的文本片段，人工标注正确的抽取结果

### Q7：修改 site-packages 后升级 GraphRAG 导致修改丢失

**解决**：
- 将修改后的文件备份到项目目录
- 升级后重新应用修改
- 或者：使用 `settings.yaml` 的 `prompt` 字段直接指向手动编写的 prompt，不依赖 prompt-tune

### Q8：实体数量太少，召回率不够

**解决**：
- 检查是否过度限制了抽取（Source Fidelity Rules 是否太严格）
- 检查 `max_gleanings` 设置（默认 1，可以增加到 2-3）
- 检查 chunk 大小（太小会导致上下文不足，太大会导致注意力分散）
- 注意：召回率是最后优先级，不要为了多抽而引入噪声

### Q9：社区报告质量差

**解决**：
- 检查 `community_report_graph.txt` 的角色描述是否准确
- 检查 REPORT RATING 标准是否合理
- 检查 `max_input_length`（默认 8000，如果社区很大可能需要增加）
- 检查 `max_length`（默认 2000，如果报告被截断需要增加）

### Q10：中文实体名称被翻译成英文

**解决**：
- 在 prompt 中明确"保留原文中的中文名称"
- 在 Output Protocol 中强调"不得使用中文全角括号"（暗示使用中文）
- 在示例中使用中文实体名称
- 在 `settings.yaml` 中设置 `language: Chinese`

---

## 10. 检查清单

在提交优化后的 prompt 之前，逐项检查：

### extract_graph.txt

- [ ] 开头有质量优先级排序
- [ ] 有源忠实性规则（列出禁止使用的信息源）
- [ ] 有实体命名规则（数值、句子、占位词、强制分类）
- [ ] 每个实体类型都有边界定义（定义 + 正面示例 + 反面示例）
- [ ] 易混淆的类型对有交叉引用
- [ ] 有描述规则（禁止宣传性语言）
- [ ] 有定量绑定规则
- [ ] 有关系规则（共现禁止、外部推断禁止、端点存在性、方向、强度）
- [ ] 关系强度有明确的评分标准（1-5 不输出）
- [ ] 有输出协议（格式、半角字符、禁止额外输出）
- [ ] 有 2-3 个真实领域示例
- [ ] 示例覆盖了典型场景（参数绑定、关系方向、不抽取的情况）
- [ ] 示例中的实体类型与 settings.yaml 中的 entity_types 一致
- [ ] 总长度在 8000 tokens 以内

### summarize_descriptions.txt

- [ ] 注入了领域角色
- [ ] 角色描述准确反映了领域
- [ ] 官方模板的其余部分保持不变

### community_report_graph.txt

- [ ] 注入了领域角色
- [ ] REPORT RATING 标准是领域特定的（多维度）
- [ ] 1 分和 10 分有明确定义
- [ ] JSON 输出格式保持不变
- [ ] Grounding Rules 保持不变

### 元 Prompt（如果优化了）

- [ ] entity_relationship.py 的规则与 extract_graph.txt 一致
- [ ] entity_types.py 强制使用目标语言
- [ ] 示例使用领域文本（不是玩具文本）
- [ ] 修改后的文件已备份到项目目录

### settings.yaml

- [ ] `prompt` 字段指向优化后的 prompt 路径
- [ ] `entity_types` 列表与 prompt 中的类型一致
- [ ] `max_gleanings` 设置合理（1-3）
- [ ] `chunking.size` 和 `chunking.overlap` 设置合理

### 验证

- [ ] 自动化检查通过（无数值实体、无句子实体、无弱关系）
- [ ] 人工抽样通过（实体名称稳定、类型正确、描述忠实）
- [ ] A/B 对比显示改善（噪声减少、类型一致性提高）
- [ ] 优化后的 prompt 已冻结到 `frozen_vN/` 目录

---

## 附录 A：官方 Prompt 与优化 Prompt 对比

| 维度 | 官方默认 | 优化后 |
|------|---------|--------|
| 质量优先级 | 无 | 6 级优先级，召回率最后 |
| 源忠实性 | "identify all entities" | 8 条禁止信息源 + 12 条禁止补充内容 |
| 实体命名 | "Name of the entity, capitalized" | 4 条子规则 + Bad/Good 示例 |
| 类型边界 | 只列出类型名称 | 每类型 5-15 行定义 + 示例 + 反面示例 |
| 关系规则 | "clearly related" | 7 条子规则（共现禁止、表格结构禁止、外部推断禁止等） |
| 关系强度 | "1 to 10" | 明确评分标准，1-5 不输出 |
| 输出格式 | "Use ## as delimiter" | 完整输出协议（半角字符、禁止 Markdown 等） |
| 示例 | 自动生成的玩具示例 | 2-3 个真实领域示例 |
| 描述规则 | 无 | 禁止宣传性语言 + 定量绑定 |
| 领域角色 | 通用 | 领域特定 |
| 评分标准 | 通用 impact severity | 领域特定多维度 |

## 附录 B：Token 预算参考

| 部分 | 建议 Token 数 |
|------|-------------|
| Goal + 质量优先级 | 100-150 |
| Entity Extraction 格式 | 100-150 |
| Source Fidelity Rules | 200-300 |
| Entity Name Rules | 300-400 |
| Type Boundary Rules (20 类型) | 1500-2500 |
| Description Rules + 定量绑定 | 200-300 |
| Relationship Rules | 400-600 |
| Output Protocol | 150-200 |
| Examples (2-3 个) | 1000-2000 |
| **总计** | **4000-6600** |

加上 `input_text`（通常 500-1200 tokens），总输入约 5000-8000 tokens，在大多数模型的上下文窗口内。

## 附录 C：参考文件

| 文件 | 说明 |
|------|------|
| `graphrag/prompts/frozen_v2/extract_graph.txt` | 优化后的实体抽取 prompt（完整示例） |
| `graphrag/prompts/frozen_v2/summarize_descriptions.txt` | 优化后的描述聚合 prompt |
| `graphrag/prompts/frozen_v2/community_report_graph.txt` | 优化后的社区报告 prompt |
| `graphrag/settings.yaml` | 配置示例（prompt 路径、entity_types） |
| `<env>/site-packages/graphrag/prompt_tune/prompt/entity_relationship.py` | 优化后的元 prompt（实体关系） |
| `<env>/site-packages/graphrag/prompt_tune/prompt/entity_types.py` | 优化后的元 prompt（实体类型） |
| `<env>/site-packages/graphrag/prompt_tune/template/extract_graph.py` | 官方 prompt 模板（对照用） |
