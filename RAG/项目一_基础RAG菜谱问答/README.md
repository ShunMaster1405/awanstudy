# 项目一：基础RAG菜谱问答系统

## 项目概述
基于HowToCook项目的菜谱数据，构建一个智能的食谱问答系统。用户可以：
- 询问具体菜品的制作方法："宫保鸡丁怎么做？"
- 寻求菜品推荐："推荐几个简单的素菜"
- 获取食材信息："红烧肉需要什么食材？"

## 技术架构
1. 数据准备：加载和预处理Markdown菜谱文件
2. 文本分块：按标题层级进行结构化分块
3. 向量索引：使用FAISS构建向量索引
4. 检索生成：基于检索的问答生成

## 项目结构
```
项目一_基础RAG菜谱问答/
├── README.md
├── requirements.txt
├── config.py
├── main.py
├── data/
│   └── recipes/          # 菜谱数据
├── rag_modules/
│   ├── __init__.py
│   ├── data_preparation.py
│   ├── index_construction.py
│   ├── retrieval_optimization.py
│   └── generation_integration.py
└── vector_index/         # 向量索引缓存（自动生成）
```

## 快速开始
```bash
# 安装依赖
pip3 install -r requirements.txt

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
build_system() - 构建RAG系统
    ├── DataPreparation().prepare_data() - 数据准备
    ├── IndexConstruction().build_index() - 索引构建
    └── GenerationIntegration() - 生成器初始化
    ↓
根据命令行参数选择运行模式：
    ├── 交互模式 (默认)
    ├── 单问题模式 (--question)
    └── 批量模式 (--file)
```

### 2. 核心模块执行流程

#### 数据准备模块 (DataPreparation)
```
prepare_data()
    ├── load_recipes() - 加载菜谱文件
    │   ├── 检查data/recipes目录
    │   ├── 如果目录为空，创建示例菜谱
    │   └── 使用DirectoryLoader加载Markdown文件
    └── split_documents() - 文本分块
        ├── 使用RecursiveCharacterTextSplitter
        ├── 按标题层级分割 (##, #, \n\n等)
        └── 添加元数据 (chunk_id, source)
```

#### 索引构建模块 (IndexConstruction)
```
build_index(documents, force_rebuild=False)
    ├── 检查vector_index/index.faiss是否存在
    ├── 如果存在且force_rebuild=False，加载现有索引
    └── 如果不存在或加载失败，创建新索引
        ├── 使用HuggingFaceEmbeddings加载嵌入模型
        ├── 使用FAISS.from_documents()创建向量存储
        └── 保存到vector_index/目录
```

#### 生成集成模块 (GenerationIntegration)
```
qa_pipeline(question, retriever)
    ├── retriever(question) - 检索相关文档
    ├── format_context(documents) - 格式化上下文
    └── generate_answer(question, documents) - 生成答案
        ├── 使用ChatOpenAI连接DeepSeek API
        ├── 应用提示模板
        └── 返回生成的答案
```

### 3. 关键函数说明

#### main.py 关键函数
- `setup_environment()`: 验证环境配置，创建必要目录
  - 入参：无
  - 出参：无
  - 功能：检查DEEPSEEK_API_KEY，创建data/recipes和vector_index目录

- `build_system()`: 构建完整的RAG系统
  - 入参：无
  - 出参：`(retriever, generator)` 元组
  - 功能：初始化所有模块，返回检索器和生成器

- `interactive_mode(retriever, generator)`: 交互式问答模式
  - 入参：`retriever` (检索函数), `generator` (生成器对象)
  - 出参：无
  - 功能：循环接收用户输入，执行问答并显示结果

#### data_preparation.py 关键函数
- `prepare_data()`: 完整的数据准备流程
  - 入参：无
  - 出参：`List[Document]` 分割后的文档块列表
  - 功能：加载菜谱文件并分割为文本块

- `load_recipes()`: 加载菜谱数据
  - 入参：无
  - 出参：`List[Document]` 原始文档列表
  - 功能：从data/recipes目录加载Markdown文件

- `split_documents(documents)`: 分割文档
  - 入参：`List[Document]` 原始文档列表
  - 出参：`List[Document]` 分割后的文档块列表
  - 功能：按配置的分块大小和重叠度分割文档

#### index_construction.py 关键函数
- `build_index(documents, force_rebuild=False)`: 构建向量索引
  - 入参：`documents` (文档列表), `force_rebuild` (是否强制重建)
  - 出参：`FAISS` 向量存储对象
  - 功能：创建或加载FAISS向量索引

- `search_similar(query, k=None)`: 搜索相似文档
  - 入参：`query` (查询字符串), `k` (返回结果数量)
  - 出参：`List[Document]` 相关文档列表
  - 功能：在向量索引中搜索与查询最相似的文档

#### generation_integration.py 关键函数
- `qa_pipeline(question, retriever)`: 完整的问答管道
  - 入参：`question` (用户问题), `retriever` (检索函数)
  - 出参：`str` 生成的答案
  - 功能：执行检索-生成完整流程

- `generate_answer(question, documents)`: 生成答案
  - 入参：`question` (用户问题), `documents` (检索到的文档)
  - 出参：`str` 生成的答案
  - 功能：使用LLM基于检索到的文档生成答案

### 4. 命令行参数说明
```bash
# 交互模式 (默认)
python3 main.py

# 单问题模式
python3 main.py --question "宫保鸡丁怎么做？"

# 批量模式
python3 main.py --file questions.txt

# 重新构建索引
python3 main.py --rebuild
```

### 5. 配置文件说明 (config.py)
- `DEEPSEEK_API_KEY`: DeepSeek API密钥
- `DEEPSEEK_API_BASE`: API基础地址 (默认: https://api.deepseek.com/v1)
- `DEEPSEEK_MODEL`: 使用的模型 (默认: deepseek-chat)
- `EMBEDDING_MODEL`: 嵌入模型 (默认: BAAI/bge-small-zh-v1.5)
- `VECTOR_STORE_PATH`: 向量索引存储路径 (默认: vector_index)
- `DATA_PATH`: 菜谱数据路径 (默认: data/recipes)
- `CHUNK_SIZE`: 文本分块大小 (默认: 500)
- `CHUNK_OVERLAP`: 分块重叠大小 (默认: 100)
- `RETRIEVAL_TOP_K`: 检索返回的文档数量 (默认: 3)
- `MAX_TOKENS`: 生成的最大token数 (默认: 1000)
- `TEMPERATURE`: 生成温度 (默认: 0.7)

## 功能特点
- 支持父子文本块策略："小块检索，大块生成"
- 高度结构化的Markdown数据处理
- 完整的RAG流程实现
- 支持多种运行模式：交互式、单问题、批量处理
- 自动创建示例数据，便于快速测试
- 向量索引缓存，避免重复构建
