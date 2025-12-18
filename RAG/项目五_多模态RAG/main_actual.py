#!/usr/bin/env python3
"""
多模态RAG系统 - 实际功能版本
"""

import os
import sys
import argparse
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 导入自定义模块
from multimodal_modules.document_processor import MultimodalDocumentProcessor
from multimodal_modules.multimodal_retriever import MultimodalRetriever
from multimodal_modules.answer_generator import MultimodalAnswerGenerator

class Config:
    """项目配置类"""
    
    # DeepSeek配置
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "yourskindeepseek")
    DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
    
    # 多模态嵌入模型配置
    MULTIMODAL_EMBEDDING_MODEL = "openai/clip-vit-base-patch32"
    
    # 数据路径
    IMAGE_DATA_PATH = "data/images"
    TEXT_DATA_PATH = "data/texts"
    MULTIMODAL_DATA_PATH = "data/multimodal"
    
    # 向量数据库配置
    VECTOR_STORE_PATH = "vector_index"
    
    # 检索配置
    RETRIEVAL_TOP_K = 3
    
    # 生成配置
    TEMPERATURE = 0.7
    MAX_TOKENS = 1000
    
    @classmethod
    def validate(cls):
        """验证配置"""
        if not cls.DEEPSEEK_API_KEY:
            raise ValueError("请设置DEEPSEEK_API_KEY环境变量")
        
        # 创建必要的目录
        os.makedirs(cls.IMAGE_DATA_PATH, exist_ok=True)
        os.makedirs(cls.TEXT_DATA_PATH, exist_ok=True)
        os.makedirs(cls.MULTIMODAL_DATA_PATH, exist_ok=True)
        os.makedirs(cls.VECTOR_STORE_PATH, exist_ok=True)
        
        print("配置验证通过")

def setup_environment():
    """设置环境"""
    print("=" * 60)
    print("多模态RAG系统 - 实际功能版本")
    print("=" * 60)
    
    # 验证配置
    try:
        Config.validate()
        print("✅ 环境配置验证通过")
    except ValueError as e:
        print(f"❌ 配置错误: {e}")
        print("请设置必要的环境变量")
        sys.exit(1)

def initialize_system():
    """初始化系统组件"""
    print("\n初始化系统组件...")
    
    # 初始化多模态文档处理器
    print("1. 初始化多模态文档处理器...")
    doc_processor = MultimodalDocumentProcessor()
    
    # 初始化多模态检索器
    print("2. 初始化多模态检索器...")
    retriever = MultimodalRetriever()
    
    # 初始化多模态答案生成器
    print("3. 初始化多模态答案生成器...")
    answer_generator = MultimodalAnswerGenerator()
    
    print("✅ 系统组件初始化完成")
    return doc_processor, retriever, answer_generator

def process_documents(doc_processor):
    """处理多模态文档"""
    print("\n" + "=" * 60)
    print("多模态文档处理阶段")
    print("=" * 60)
    
    # 处理文档管道
    documents = doc_processor.process_pipeline()
    
    if not documents:
        print("❌ 文档处理失败，没有可用的文档")
        return None
    
    print(f"✅ 文档处理完成: {len(documents)} 个文档块")
    return documents

def build_index(retriever, documents):
    """构建多模态索引"""
    print("\n" + "=" * 60)
    print("多模态索引构建阶段")
    print("=" * 60)
    
    if not documents:
        print("❌ 没有文档可用于构建索引")
        return False
    
    # 构建向量存储
    retriever.build_vector_store(documents)
    
    print("✅ 索引构建完成")
    return True

def load_index(retriever):
    """加载索引"""
    print("\n尝试加载现有索引...")
    
    if retriever.load_vector_store():
        print("✅ 索引加载成功")
        return True
    else:
        print("❌ 索引加载失败，需要重新构建")
        return False

def run_query_pipeline(retriever, answer_generator, query, image_path=None):
    """运行多模态查询管道"""
    print("\n" + "=" * 60)
    print("多模态查询处理管道")
    print("=" * 60)
    
    print(f"查询: {query}")
    if image_path:
        print(f"图像: {image_path}")
    
    # 执行多模态检索
    print("\n执行多模态检索...")
    retrieval_stats = retriever.multimodal_retrieval(query, image_path)
    
    if not retrieval_stats.get("success", False):
        print("❌ 检索失败")
        return None
    
    # 获取最终文档
    final_docs = []
    if retrieval_stats.get("final_docs", 0) > 0:
        # 从检索器获取文档
        final_docs = retriever.retrieve_pipeline(query, image_path)
    
    if not final_docs:
        print("❌ 没有检索到相关文档")
        return None
    
    # 生成多模态答案
    print("\n生成多模态答案...")
    answer_result = answer_generator.generate_pipeline(query, final_docs, retrieval_stats, image_path)
    
    return answer_result

def demo_queries(retriever, answer_generator):
    """演示多模态查询"""
    print("\n" + "=" * 60)
    print("多模态演示查询")
    print("=" * 60)
    
    demo_queries = [
        ("什么是多模态学习？", None),
        ("图像识别技术有哪些应用？", None),
        ("如何实现文本到图像的检索？", None),
        ("多模态RAG与传统RAG有什么区别？", None),
        ("视觉问答系统的工作原理是什么？", None)
    ]
    
    for i, (query, image_path) in enumerate(demo_queries, 1):
        print(f"\n[{i}/{len(demo_queries)}] 查询: {query}")
        
        result = run_query_pipeline(retriever, answer_generator, query, image_path)
        
        if result and result.get("success", False):
            print(f"\n答案: {result['answer'][:200]}...")
            eval_score = result['evaluation'].get('overall_score', 0)
            print(f"答案质量评分: {eval_score}/5")
        else:
            print("❌ 查询处理失败")
        
        print("-" * 40)

def interactive_mode(doc_processor, retriever, answer_generator):
    """交互模式"""
    print("\n" + "=" * 60)
    print("多模态交互模式")
    print("输入 'quit' 或 'exit' 退出")
    print("=" * 60)
    
    # 检查是否需要处理文档
    if not retriever.vector_store:
        print("⚠️  向量存储未初始化，需要先处理文档")
        documents = process_documents(doc_processor)
        if documents:
            build_index(retriever, documents)
        else:
            print("❌ 无法初始化系统，退出交互模式")
            return
    
    while True:
        try:
            query = input("\n请输入问题 (输入 'help' 查看帮助): ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("退出系统")
                break
            
            if query.lower() == 'help':
                print("\n可用命令:")
                print("  help      - 显示帮助信息")
                print("  demo      - 运行演示查询")
                print("  process   - 重新处理文档")
                print("  rebuild   - 重新构建索引")
                print("  quit      - 退出系统")
                print("\n或者直接输入问题进行查询")
                continue
            
            if query.lower() == 'demo':
                demo_queries(retriever, answer_generator)
                continue
            
            if query.lower() == 'process':
                documents = process_documents(doc_processor)
                if documents:
                    build_index(retriever, documents)
                continue
            
            if query.lower() == 'rebuild':
                documents = process_documents(doc_processor)
                if documents:
                    build_index(retriever, documents)
                continue
            
            if query:
                # 运行查询管道
                result = run_query_pipeline(retriever, answer_generator, query)
                
                if result and result.get("success", False):
                    print("\n" + "=" * 60)
                    print("查询结果")
                    print("=" * 60)
                    
                    print(f"\n问题: {result['question']}")
                    print(f"\n答案:\n{result['answer']}")
                    
                    # 显示评估结果
                    eval_result = result['evaluation']
                    print(f"\n答案质量评估:")
                    print(f"  总体评分: {eval_result.get('overall_score', 0)}/5")
                    
                    # 显示维度评分
                    dimensions = eval_result.get('dimensions', {})
                    if dimensions:
                        print("  维度评分:")
                        for dim, score in dimensions.items():
                            print(f"    {dim}: {score}/5")
                    
                    # 显示统计信息
                    print(f"\n统计信息:")
                    print(f"  使用文档数: {result['documents_used']}")
                    
                    if 'retrieval_stats' in result:
                        print(f"  检索统计:\n{result['retrieval_stats']}")
                    
                else:
                    print("❌ 无法生成答案")
            
        except KeyboardInterrupt:
            print("\n\n用户中断，退出系统")
            break
        except Exception as e:
            print(f"❌ 错误: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="多模态RAG系统 - 实际功能版本")
    parser.add_argument("--demo", "-d", action="store_true", help="运行演示模式")
    parser.add_argument("--rebuild", "-r", action="store_true", help="重新构建索引")
    
    args = parser.parse_args()
    
    # 设置环境
    setup_environment()
    
    # 初始化系统组件
    doc_processor, retriever, answer_generator = initialize_system()
    
    # 处理文档和构建索引
    if args.rebuild or not load_index(retriever):
        print("\n需要构建新的索引...")
        documents = process_documents(doc_processor)
        if documents:
            build_index(retriever, documents)
        else:
            print("❌ 无法构建索引，退出系统")
            sys.exit(1)
    
    if args.demo:
        # 演示模式
        demo_queries(retriever, answer_generator)
    else:
        # 交互模式
        interactive_mode(doc_processor, retriever, answer_generator)

if __name__ == "__main__":
    main()
