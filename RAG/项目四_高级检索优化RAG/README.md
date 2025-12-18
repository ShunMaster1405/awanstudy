# 项目四：高级检索优化RAG系统

## 项目概述
集成多种高级检索优化技术的RAG系统，包括重排序（Re-ranking）、压缩（Compression）和校正（Corrective-RAG）等技术，显著提升检索精度和生成质量。

## 技术架构
1. **重排序技术**：Cross-Encoder、ColBERT、RankLLM等
2. **压缩技术**：上下文压缩、文档过滤、内容提取
3. **校正技术**：Corrective-RAG（C-RAG）检索质量评估
4. **管道组合**：DocumentCompressorPipeline多处理器组合

## 核心特性
- **多级重排序**：从粗排到精排的多级优化
- **智能压缩**：基于LLM的上下文压缩和过滤
- **自我校正**：检索质量评估和外部知识补充
- **模块化设计**：灵活组合不同优化技术

## 项目结构
```
项目四_高级检索优化RAG/
├── README.md
├── requirements.txt
├── config.py
├── main.py
├── data/
│   └── documents/          # 文档数据
├── advanced_modules/
│   ├── __init__.py
│   ├── reranking.py        # 重排序模块
│   ├── compression.py      # 压缩模块
│   ├── correction.py       # 校正模块
│   ├── pipeline.py         # 管道组合
│   └── evaluation.py       # 评估模块
└── models/                 # 预训练模型
```

## 快速开始
```bash
# 安装依赖
pip3 install -r requirements.txt

# 下载预训练模型
python3 download_models.py

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
    │   ├── reranking_demo() - 重排序演示
    │   ├── compression_demo() - 压缩技术演示
    │   ├── correction_demo() - 校正技术演示
    │   └── pipeline_demo() - 管道组合演示
    └── 交互模式 (默认)
        └── interactive_mode() - 交互式命令界面
```

### 2. 核心模块执行流程

#### 环境设置模块
```
setup_environment()
    ├── Config.validate() - 验证配置
    │   ├── 检查DEEPSEEK_API_KEY环境变量
    │   ├── 创建data/documents目录
    │   └── 创建vector_index目录
    └── 打印系统标题和配置状态
```

#### 示例数据创建模块
```
create_sample_data()
    ├── 定义示例文档数据（检索优化技术相关）
    ├── 创建data/documents目录
    └── 将示例数据保存为文本文件
```

#### 重排序演示模块
```
reranking_demo()
    ├── Cross-Encoder: 对查询-文档对进行联合编码和评分
    ├── ColBERT: 基于上下文的晚期交互模型
    ├── RankLLM: 使用LLM进行零样本重排序
    └── MonoT5: 基于T5的序列到序列重排序
```

#### 压缩技术演示模块
```
compression_demo()
    ├── 文档过滤: 基于相关性过滤文档
    ├── 内容提取: 提取文档中的关键信息
    ├── 摘要生成: 生成文档摘要
    └── 语义压缩: 基于语义的压缩和重组
```

#### 校正技术演示模块
```
correction_demo()
    ├── 检索质量评估: 评估检索结果的相关性和完整性
    ├── 外部知识补充: 当检索质量不足时补充外部知识
    ├── 检索结果校正: 校正和优化检索结果
    └── 增强生成: 基于校正后的结果进行生成
```

#### 管道组合演示模块
```
pipeline_demo()
    ├── 基础检索器: 初步检索相关文档
    ├── 重排序器: 对检索结果进行重排序
    ├── 压缩器: 压缩和过滤文档内容
    ├── 校正器: 评估和校正检索质量
    └── 生成器: 基于优化后的结果生成答案
```

### 3. 关键函数说明

#### main.py 关键函数
- `setup_environment()`: 验证环境配置，创建必要目录
  - 入参：无
  - 出参：无
  - 功能：检查DEEPSEEK_API_KEY，创建data/documents和vector_index目录

- `create_sample_data()`: 创建示例数据
  - 入参：无
  - 出参：无
  - 功能：在data/documents目录中创建示例文档文件

- `reranking_demo()`: 重排序演示
  - 入参：无
  - 出参：无
  - 功能：展示各种重排序技术的原理和应用

- `compression_demo()`: 压缩技术演示
  - 入参：无
  - 出参：无
  - 功能：展示上下文压缩技术的实现方法

- `correction_demo()`: 校正技术演示
  - 入参：无
  - 出参：无
  - 功能：展示Corrective-RAG的校正流程

- `pipeline_demo()`: 管道组合演示
  - 入参：无
  - 出参：无
  - 功能：展示多处理器管道组合的优化流程

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
- `DEEPSEEK_API_KEY`: DeepSeek API密钥
- `DEEPSEEK_MODEL`: 使用的模型 (默认: deepseek-chat)
- `EMBEDDING_MODEL`: 嵌入模型 (默认: BAAI/bge-small-zh-v1.5)
- `DATA_PATH`: 文档数据路径 (默认: data/documents)
- `VECTOR_STORE_PATH`: 向量索引存储路径 (默认: vector_index)
- `RERANKER_MODEL`: 重排序模型 (默认: BAAI/bge-reranker-large)
- `RETRIEVAL_TOP_K`: 初步检索返回结果数量 (默认: 10)
- `RERANK_TOP_K`: 重排序后返回结果数量 (默认: 3)

## 应用场景
- 高精度问答：对答案准确性要求高的场景
- 专业领域：医疗、法律、金融等专业领域
- 生产环境：需要稳定可靠检索的系统
- 复杂查询：需要多步推理的复杂问题
