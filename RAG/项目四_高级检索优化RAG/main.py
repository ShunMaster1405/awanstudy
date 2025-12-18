#!/usr/bin/env python3
"""
高级检索优化RAG系统主程序
"""

import os
import sys
import argparse
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class Config:
    """项目配置类"""
    
    # DeepSeek配置
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "yourskindeepseek")
    DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    
    # 嵌入模型配置
    EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
    
    # 数据路径
    DATA_PATH = "data/documents"
    
    # 向量数据库配置
    VECTOR_STORE_PATH = "vector_index"
    
    # 重排序模型配置
    RERANKER_MODEL = "BAAI/bge-reranker-large"
    
    # 检索配置
    RETRIEVAL_TOP_K = 10
    RERANK_TOP_K = 3
    
    @classmethod
    def validate(cls):
        """验证配置"""
        if not cls.DEEPSEEK_API_KEY:
            raise ValueError("请设置DEEPSEEK_API_KEY环境变量")
        
        # 创建必要的目录
        os.makedirs(cls.DATA_PATH, exist_ok=True)
        os.makedirs(cls.VECTOR_STORE_PATH, exist_ok=True)
        
        print("配置验证通过")

def setup_environment():
    """设置环境"""
    print("=" * 50)
    print("高级检索优化RAG系统")
    print("=" * 50)
    
    # 验证配置
    try:
        Config.validate()
        print("环境配置验证通过")
    except ValueError as e:
        print(f"配置错误: {e}")
        print("请设置必要的环境变量")
        sys.exit(1)

def create_sample_data():
    """创建示例数据"""
    print("\n创建示例文档数据...")
    
    sample_documents = [
        {
            "title": "重排序技术",
            "content": "重排序技术通过重新排序初步检索结果来提升检索精度，常用方法包括Cross-Encoder、ColBERT和RankLLM等。"
        },
        {
            "title": "上下文压缩",
            "content": "上下文压缩技术通过过滤和提取关键信息来减少输入长度，提高生成效率和质量。"
        },
        {
            "title": "Corrective-RAG",
            "content": "Corrective-RAG（C-RAG）通过评估检索质量并补充外部知识来校正检索结果。"
        },
        {
            "title": "多级优化",
            "content": "多级优化结合多种技术，从粗排到精排逐步提升检索性能。"
        }
    ]
    
    os.makedirs(Config.DATA_PATH, exist_ok=True)
    
    for i, doc in enumerate(sample_documents, 1):
        filename = f"doc_{i}.txt"
        filepath = os.path.join(Config.DATA_PATH, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# {doc['title']}\n\n")
            f.write(doc['content'])
        
        print(f"创建文件: {filename}")
    
    print(f"创建了 {len(sample_documents)} 个示例文档")

def reranking_demo():
    """重排序演示"""
    print("\n" + "=" * 50)
    print("重排序技术演示")
    print("=" * 50)
    
    print("常用重排序技术:")
    print("1. Cross-Encoder: 对查询-文档对进行联合编码和评分")
    print("2. ColBERT: 基于上下文的晚期交互模型")
    print("3. RankLLM: 使用LLM进行零样本重排序")
    print("4. MonoT5: 基于T5的序列到序列重排序")

def compression_demo():
    """压缩技术演示"""
    print("\n" + "=" * 50)
    print("压缩技术演示")
    print("=" * 50)
    
    print("上下文压缩技术:")
    print("1. 文档过滤: 基于相关性过滤文档")
    print("2. 内容提取: 提取文档中的关键信息")
    print("3. 摘要生成: 生成文档摘要")
    print("4. 语义压缩: 基于语义的压缩和重组")

def correction_demo():
    """校正技术演示"""
    print("\n" + "=" * 50)
    print("校正技术演示")
    print("=" * 50)
    
    print("Corrective-RAG (C-RAG) 流程:")
    print("1. 检索质量评估: 评估检索结果的相关性和完整性")
    print("2. 外部知识补充: 当检索质量不足时补充外部知识")
    print("3. 检索结果校正: 校正和优化检索结果")
    print("4. 增强生成: 基于校正后的结果进行生成")

def pipeline_demo():
    """管道组合演示"""
    print("\n" + "=" * 50)
    print("管道组合演示")
    print("=" * 50)
    
    print("DocumentCompressorPipeline 组合:")
    print("1. 基础检索器: 初步检索相关文档")
    print("2. 重排序器: 对检索结果进行重排序")
    print("3. 压缩器: 压缩和过滤文档内容")
    print("4. 校正器: 评估和校正检索质量")
    print("5. 生成器: 基于优化后的结果生成答案")

def interactive_mode():
    """交互模式"""
    print("\n" + "=" * 50)
    print("交互模式")
    print("输入 'quit' 或 'exit' 退出")
    print("=" * 50)
    
    while True:
        try:
            command = input("\n请输入命令 (help查看帮助): ").strip()
            
            if command.lower() in ['quit', 'exit', 'q']:
                print("退出系统")
                break
            
            if command.lower() == 'help':
                print("\n可用命令:")
                print("  help      - 显示帮助信息")
                print("  data      - 创建示例数据")
                print("  rerank    - 重排序演示")
                print("  compress  - 压缩技术演示")
                print("  correct   - 校正技术演示")
                print("  pipeline  - 管道组合演示")
                print("  quit      - 退出系统")
            
            elif command.lower() == 'data':
                create_sample_data()
            
            elif command.lower() == 'rerank':
                reranking_demo()
            
            elif command.lower() == 'compress':
                compression_demo()
            
            elif command.lower() == 'correct':
                correction_demo()
            
            elif command.lower() == 'pipeline':
                pipeline_demo()
            
            elif command:
                print(f"未知命令: {command}")
                print("输入 'help' 查看可用命令")
            
        except KeyboardInterrupt:
            print("\n\n用户中断，退出系统")
            break
        except Exception as e:
            print(f"错误: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="高级检索优化RAG系统")
    parser.add_argument("--demo", "-d", action="store_true", help="运行演示模式")
    
    args = parser.parse_args()
    
    # 设置环境
    setup_environment()
    
    if args.demo:
        # 演示模式
        create_sample_data()
        reranking_demo()
        compression_demo()
        correction_demo()
        pipeline_demo()
    else:
        # 交互模式
        interactive_mode()

if __name__ == "__main__":
    main()
