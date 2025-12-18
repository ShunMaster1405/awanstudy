#!/usr/bin/env python3
"""
基础RAG菜谱问答系统主程序
"""

import sys
import argparse
from config import Config
from rag_modules import DataPreparation, IndexConstruction, GenerationIntegration

def setup_environment():
    """设置环境"""
    print("=" * 50)
    print("基础RAG菜谱问答系统")
    print("=" * 50)
    
    # 验证配置
    try:
        Config.validate()
        print("环境配置验证通过")
    except ValueError as e:
        print(f"配置错误: {e}")
        print("请设置DEEPSEEK_API_KEY环境变量")
        sys.exit(1)

def build_system():
    """构建RAG系统"""
    print("\n" + "=" * 50)
    print("构建RAG系统")
    print("=" * 50)
    
    # 数据准备
    data_prep = DataPreparation()
    documents = data_prep.prepare_data()
    
    # 索引构建
    index_constructor = IndexConstruction()
    vector_store = index_constructor.build_index(documents)
    
    # 创建检索器
    retriever = lambda query: index_constructor.search_similar(query)
    
    # 生成集成
    generator = GenerationIntegration()
    
    return retriever, generator

def interactive_mode(retriever, generator):
    """交互模式"""
    print("\n" + "=" * 50)
    print("进入交互模式")
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
            
            # 执行问答
            answer = generator.qa_pipeline(question, retriever)
            
            print("\n" + "-" * 50)
            print("答案:")
            print(answer)
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n\n用户中断，退出系统")
            break
        except Exception as e:
            print(f"错误: {e}")

def batch_mode(retriever, generator, questions):
    """批量模式"""
    print("\n" + "=" * 50)
    print("批量处理模式")
    print("=" * 50)
    
    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] 问题: {question}")
        
        try:
            answer = generator.qa_pipeline(question, retriever)
            print(f"答案: {answer[:200]}...")
        except Exception as e:
            print(f"处理失败: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="基础RAG菜谱问答系统")
    parser.add_argument("--question", "-q", type=str, help="单个问题")
    parser.add_argument("--file", "-f", type=str, help="包含问题的文件")
    parser.add_argument("--rebuild", "-r", action="store_true", help="重新构建索引")
    
    args = parser.parse_args()
    
    # 设置环境
    setup_environment()
    
    # 构建系统
    retriever, generator = build_system()
    
    # 根据参数选择模式
    if args.question:
        # 单个问题模式
        answer = generator.qa_pipeline(args.question, retriever)
        print("\n" + "=" * 50)
        print(f"问题: {args.question}")
        print(f"答案: {answer}")
        print("=" * 50)
    
    elif args.file:
        # 批量模式
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                questions = [line.strip() for line in f if line.strip()]
            
            batch_mode(retriever, generator, questions)
        except FileNotFoundError:
            print(f"文件不存在: {args.file}")
            sys.exit(1)
    
    else:
        # 交互模式
        interactive_mode(retriever, generator)

if __name__ == "__main__":
    main()
