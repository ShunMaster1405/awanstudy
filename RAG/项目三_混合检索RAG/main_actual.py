#!/usr/bin/env python3
"""
混合检索RAG系统 - 实际实现
"""

import os
import sys
import argparse
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 导入配置
from config import Config

# 导入模块
from hybrid_modules.document_processor import DocumentProcessor
from hybrid_modules.hybrid_index import HybridIndex
from hybrid_modules.hybrid_retriever import HybridRetriever
from hybrid_modules.answer_generator import AnswerGenerator

def setup_environment():
    """设置环境"""
    print("=" * 50)
    print("混合检索RAG系统 - 实际实现")
    print("=" * 50)
    
    # 验证配置
    try:
        Config.validate()
        print("环境配置验证通过")
    except ValueError as e:
        print(f"配置错误: {e}")
        print("请设置必要的环境变量")
        sys.exit(1)

def build_hybrid_rag_system():
    """构建混合检索RAG系统"""
    print("\n" + "=" * 50)
    print("构建混合检索RAG系统")
    print("=" * 50)
    
    # 1. 文档处理
    print("\n1. 文档处理...")
    processor = DocumentProcessor()
    documents = processor.load_and_process_documents()
    
    if not documents:
        print("没有找到文档，创建示例文档...")
        documents = processor.create_sample_documents()
    
    print(f"文档处理完成: {len(documents)} 个文档块")
    
    # 2. 构建混合索引
    print("\n2. 构建混合索引...")
    index_builder = HybridIndex()
    index_info = index_builder.build_index(documents)
    
    print(f"混合索引构建完成:")
    print(f"  - 文档数量: {index_info['num_documents']}")
    print(f"  - 密集向量维度: {index_info.get('dense_dim', 'N/A')}")
    print(f"  - 稀疏向量特征: {index_info.get('sparse_features', 'N/A')}")
    
    # 3. 创建混合检索器
    print("\n3. 创建混合检索器...")
    retriever = HybridRetriever()
    
    # 4. 创建答案生成器
    print("\n4. 创建答案生成器...")
    generator = AnswerGenerator()
    
    print("\n混合检索RAG系统构建完成!")
    
    return {
        "processor": processor,
        "index_builder": index_builder,
        "retriever": retriever,
        "generator": generator,
        "documents": documents
    }

def ask_question(system_components, question):
    """提问并获取答案"""
    print("\n" + "=" * 50)
    print(f"问题: {question}")
    print("=" * 50)
    
    retriever = system_components["retriever"]
    generator = system_components["generator"]
    
    print("\n处理问题...")
    
    # 混合检索
    print("  - 执行混合检索...")
    retrieval_results = retriever.retrieve(question)
    
    if not retrieval_results:
        print("  - 未找到相关文档")
        return None
    
    print(f"  - 找到 {len(retrieval_results)} 个相关文档")
    
    # 生成答案
    print("  - 生成答案...")
    answer = generator.generate_answer(question, retrieval_results)
    
    return answer

def interactive_mode(system_components):
    """交互模式"""
    print("\n" + "=" * 50)
    print("交互模式")
    print("输入 'quit' 或 'exit' 退出")
    print("=" * 50)
    
    while True:
        try:
            question = input("\n请输入问题: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("退出系统")
                break
            
            if not question:
                continue
            
            # 提问并获取答案
            answer = ask_question(system_components, question)
            
            if answer:
                print("\n答案:")
                print("-" * 40)
                print(answer)
                print("-" * 40)
            else:
                print("\n无法回答该问题")
            
        except KeyboardInterrupt:
            print("\n\n用户中断，退出系统")
            break
        except Exception as e:
            print(f"\n错误: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="混合检索RAG系统 - 实际实现")
    parser.add_argument("--question", "-q", type=str, help="直接提问的问题")
    parser.add_argument("--interactive", "-i", action="store_true", help="交互模式")
    
    args = parser.parse_args()
    
    # 设置环境
    setup_environment()
    
    # 构建系统
    system_components = build_hybrid_rag_system()
    
    if args.question:
        # 直接提问模式
        answer = ask_question(system_components, args.question)
        if answer:
            print("\n答案:")
            print("-" * 40)
            print(answer)
            print("-" * 40)
        else:
            print("\n无法回答该问题")
    
    elif args.interactive:
        # 交互模式
        interactive_mode(system_components)
    
    else:
        # 默认演示模式
        print("\n" + "=" * 50)
        print("演示模式")
        print("=" * 50)
        
        # 演示问题
        demo_questions = [
            "人工智能是什么？",
            "机器学习有哪些类型？",
            "深度学习有什么应用？"
        ]
        
        for question in demo_questions:
            answer = ask_question(system_components, question)
            if answer:
                print(f"\n问题: {question}")
                print(f"答案: {answer[:100]}...")
            else:
                print(f"\n问题: {question}")
                print("答案: 无法回答")
        
        print("\n演示完成!")
        print("使用 --interactive 进入交互模式")
        print("使用 --question '你的问题' 直接提问")

if __name__ == "__main__":
    main()
