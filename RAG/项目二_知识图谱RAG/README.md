# 项目二：基于知识图谱的RAG系统

## 项目概述
基于知识图谱的RAG（KG-RAG）系统，利用Neo4j图数据库存储结构化知识，增强RAG系统的推理能力和事实准确性。

## 技术架构
1. **知识图谱构建**：从文本中抽取实体和关系，构建知识图谱
2. **图数据库存储**：使用Neo4j存储和管理知识图谱
3. **图检索**：基于Cypher查询语言进行图检索
4. **混合检索**：结合向量检索和图检索
5. **增强生成**：利用结构化知识进行推理生成

## 核心特性
- **多跳推理**：支持复杂查询的多步推理
- **事实准确性**：基于结构化知识减少幻觉
- **可解释性**：提供推理路径和证据链
- **混合检索**：结合向量相似度和图结构检索

## 项目结构
```
项目二_知识图谱RAG/
├── README.md
├── requirements.txt
├── config.py
├── main.py
├── data/
│   ├── raw_texts/          # 原始文本数据
│   └── knowledge_graph/    # 知识图谱数据
├── kg_modules/
│   ├── __init__.py
│   ├── knowledge_extraction.py    # 知识抽取
│   ├── graph_construction.py      # 图构建
│   ├── graph_retrieval.py         # 图检索
│   └── hybrid_rag.py              # 混合RAG
└── neo4j_data/             # Neo4j数据库文件
```

## 快速开始
```bash
# 安装依赖
pip3 install -r requirements.txt

# 启动Neo4j服务（需要Docker）
docker-compose up -d

# 运行项目
python3 main.py
```

## 整体执行流程

### 1. 程序启动
执行 `python3 main.py` 时，程序按以下流程运行：

```
main() 函数入口
    ↓
setup_environment() - 环境配置验证
    ↓
根据命令行参数选择运行模式：
    ├── 演示模式 (--demo)
    │   ├── create_sample_data() - 创建示例数据
    │   ├── knowledge_extraction_demo() - 知识抽取演示
    │   ├── graph_retrieval_demo() - 图检索演示
    │   └── hybrid_rag_demo() - 混合RAG演示
    └── 交互模式 (默认)
        └── interactive_mode() - 交互式命令界面
```

### 2. 核心模块执行流程

#### 环境设置模块
```
setup_environment()
    ├── Config.validate() - 验证配置
    │   ├── 检查DEEPSEEK_API_KEY环境变量
    │   └── 创建必要的目录结构
    └── 打印系统标题和配置状态
```

#### 示例数据创建模块
```
create_sample_data()
    ├── 定义示例文本数据（人工智能相关）
    ├── 创建data/raw_texts目录
    └── 将示例数据保存为文本文件
```

#### 知识抽取演示模块
```
knowledge_extraction_demo()
    ├── 实体识别：识别文档中的命名实体
    ├── 关系抽取：提取实体之间的关系
    ├── 知识图谱构建：将实体和关系存储到Neo4j
    └── 图检索：使用Cypher查询语言检索知识
```

#### 图检索演示模块
```
graph_retrieval_demo()
    ├── 展示示例Cypher查询
    ├── 实体查询：检索图中的实体
    ├── 关系查询：检索实体间的关系
    └── 路径查询：检索多跳关系路径
```

#### 混合RAG演示模块
```
hybrid_rag_demo()
    ├── 向量检索：基于语义相似度检索相关文档
    ├── 图检索：基于知识图谱检索相关实体和关系
    ├── 结果融合：结合两种检索结果
    └── 增强生成：利用结构化知识进行推理生成
```

### 3. 关键函数说明

#### main.py 关键函数
- `setup_environment()`: 验证环境配置，创建必要目录
  - 入参：无
  - 出参：无
  - 功能：检查DEEPSEEK_API_KEY，创建data/raw_texts目录

- `create_sample_data()`: 创建示例数据
  - 入参：无
  - 出参：无
  - 功能：在data/raw_texts目录中创建示例文本文件

- `knowledge_extraction_demo()`: 知识抽取演示
  - 入参：无
  - 出参：无
  - 功能：展示知识抽取的完整流程

- `graph_retrieval_demo()`: 图检索演示
  - 入参：无
  - 出参：无
  - 功能：展示Cypher查询和知识图谱检索

- `hybrid_rag_demo()`: 混合RAG演示
  - 入参：无
  - 出参：无
  - 功能：展示向量检索和图检索的融合

- `interactive_mode()`: 交互式命令界面
  - 入参：无
  - 出参：无
  - 功能：提供交互式命令界面，支持多种演示功能

### 4. 命令行参数说明
```bash
# 交互模式 (默认)
python3 main.py

# 演示模式
python3 main.py --demo
```

### 5. 配置文件说明 (config.py)
- `NEO4J_URI`: Neo4j数据库连接URI (默认: bolt://localhost:7687)
- `NEO4J_USERNAME`: Neo4j用户名 (默认: neo4j)
- `NEO4J_PASSWORD`: Neo4j密码 (默认: password)
- `DEEPSEEK_API_KEY`: DeepSeek API密钥
- `DEEPSEEK_MODEL`: 使用的模型 (默认: deepseek-chat)
- `DATA_PATH`: 原始文本数据路径 (默认: data/raw_texts)
- `EMBEDDING_MODEL`: 嵌入模型 (默认: BAAI/bge-small-zh-v1.5)

## 应用场景
- 金融领域：公司关系查询、投资分析
- 医疗领域：疾病关系、药物相互作用
- 学术研究：文献知识图谱、研究关系网络
- 企业知识管理：员工技能图谱、项目关系
