# 项目五：多模态RAG系统

## 项目概述
支持文本、图像等多模态数据的RAG系统，结合视觉和语言模型，实现跨模态检索和生成。

## 技术架构
1. **多模态嵌入**：CLIP、BLIP等多模态嵌入模型
2. **跨模态检索**：文本到图像、图像到文本的跨模态检索
3. **多模态生成**：结合视觉和文本信息的生成
4. **统一索引**：多模态数据的统一向量索引

## 核心特性
- **跨模态理解**：理解文本和图像的语义关联
- **多模态检索**：支持多种模态的混合检索
- **视觉问答**：基于图像的问答能力
- **图文生成**：结合视觉信息的文本生成

## 项目结构
```
项目五_多模态RAG/
├── README.md
├── requirements.txt
├── config.py
├── main.py
├── data/
│   ├── images/             # 图像数据
│   ├── texts/              # 文本数据
│   └── multimodal/         # 多模态数据
├── multimodal_modules/
│   ├── __init__.py
│   ├── multimodal_embedding.py  # 多模态嵌入
│   ├── cross_modal_retrieval.py # 跨模态检索
│   ├── visual_qa.py             # 视觉问答
│   └── multimodal_generation.py # 多模态生成
└── vector_index/           # 向量索引
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
    │   ├── multimodal_embedding_demo() - 多模态嵌入演示
    │   ├── cross_modal_retrieval_demo() - 跨模态检索演示
    │   ├── visual_qa_demo() - 视觉问答演示
    │   └── multimodal_generation_demo() - 多模态生成演示
    └── 交互模式 (默认)
        └── interactive_mode() - 交互式命令界面
```

### 2. 核心模块执行流程

#### 环境设置模块
```
setup_environment()
    ├── Config.validate() - 验证配置
    │   ├── 检查DEEPSEEK_API_KEY环境变量
    │   ├── 创建data/images目录
    │   ├── 创建data/texts目录
    │   ├── 创建data/multimodal目录
    │   └── 创建vector_index目录
    └── 打印系统标题和配置状态
```

#### 示例数据创建模块
```
create_sample_data()
    ├── 定义示例文本数据（多模态AI相关）
    ├── 创建data/texts目录
    ├── 将示例文本数据保存为文件
    └── 提示用户准备真实图像数据
```

#### 多模态嵌入演示模块
```
multimodal_embedding_demo()
    ├── CLIP: OpenAI的对比语言-图像预训练模型
    ├── BLIP: 引导语言-图像预训练模型
    ├── ALIGN: 大规模视觉-语言表示学习
    └── Florence: 微软的统一视觉-语言模型
```

#### 跨模态检索演示模块
```
cross_modal_retrieval_demo()
    ├── 文本到图像检索: 用文本查询检索相关图像
    ├── 图像到文本检索: 用图像查询检索相关文本
    ├── 图像到图像检索: 用图像查询检索相似图像
    └── 多模态混合检索: 结合文本和图像查询
```

#### 视觉问答演示模块
```
visual_qa_demo()
    ├── 图像理解: 分析图像内容，识别物体、场景等
    ├── 问题理解: 理解用户提出的问题
    ├── 多模态推理: 结合视觉和语言信息进行推理
    └── 答案生成: 生成基于图像内容的答案
```

#### 多模态生成演示模块
```
multimodal_generation_demo()
    ├── 图像描述: 为图像生成文字描述
    ├── 视觉故事: 基于图像生成故事
    ├── 图文问答: 结合图像和文本的问答
    └── 多模态对话: 支持图像和文本的对话
```

### 3. 关键函数说明

#### main.py 关键函数
- `setup_environment()`: 验证环境配置，创建必要目录
  - 入参：无
  - 出参：无
  - 功能：检查DEEPSEEK_API_KEY，创建多模态数据目录和vector_index目录

- `create_sample_data()`: 创建示例数据
  - 入参：无
  - 出参：无
  - 功能：在data/texts目录中创建示例文本文件，提示准备图像数据

- `multimodal_embedding_demo()`: 多模态嵌入演示
  - 入参：无
  - 出参：无
  - 功能：展示多模态嵌入模型的原理和应用

- `cross_modal_retrieval_demo()`: 跨模态检索演示
  - 入参：无
  - 出参：无
  - 功能：展示各种跨模态检索类型和应用场景

- `visual_qa_demo()`: 视觉问答演示
  - 入参：无
  - 出参：无
  - 功能：展示视觉问答的流程和示例问题

- `multimodal_generation_demo()`: 多模态生成演示
  - 入参：无
  - 出参：无
  - 功能：展示多模态生成能力和技术实现

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
- `MULTIMODAL_EMBEDDING_MODEL`: 多模态嵌入模型 (默认: openai/clip-vit-base-patch32)
- `IMAGE_DATA_PATH`: 图像数据路径 (默认: data/images)
- `TEXT_DATA_PATH`: 文本数据路径 (默认: data/texts)
- `MULTIMODAL_DATA_PATH`: 多模态数据路径 (默认: data/multimodal)
- `VECTOR_STORE_PATH`: 向量索引存储路径 (默认: vector_index)
- `RETRIEVAL_TOP_K`: 检索返回结果数量 (默认: 3)

## 应用场景
- 电商搜索：图文商品检索
- 内容审核：图像和文本联合分析
- 教育领域：图文教材问答
- 媒体分析：新闻图文内容理解
- 智能助手：多模态交互助手
