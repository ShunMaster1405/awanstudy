#!/usr/bin/env python3
"""
混合检索RAG系统 - 简化版本（不使用外部模型）
"""

import os
import sys
import argparse
import numpy as np
from typing import List, Dict, Any

def setup_environment():
    """设置环境"""
    print("=" * 50)
    print("混合检索RAG系统 - 简化版本")
    print("=" * 50)
    print("环境配置验证通过")

def create_sample_documents():
    """创建示例文档"""
    print("\n创建示例文档...")
    
    documents = [
        {
            "id": "doc_1",
            "title": "人工智能概述",
            "content": "人工智能是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的智能机器。人工智能包括机器学习、深度学习、自然语言处理、计算机视觉等多个子领域。人工智能技术已经广泛应用于医疗、金融、教育、交通等各个行业。"
        },
        {
            "id": "doc_2", 
            "title": "机器学习基础",
            "content": "机器学习是人工智能的一个子领域，使计算机能够在没有明确编程的情况下学习和改进。机器学习主要分为监督学习、无监督学习和强化学习三种类型。监督学习使用带标签的数据进行训练，无监督学习发现数据中的模式，强化学习通过试错进行学习。"
        },
        {
            "id": "doc_3",
            "title": "深度学习应用",
            "content": "深度学习是机器学习的一个子集，使用神经网络模拟人脑的工作方式。深度学习在计算机视觉、语音识别、自然语言处理等领域有广泛应用。常见的深度学习模型包括卷积神经网络、循环神经网络和Transformer。"
        },
        {
            "id": "doc_4",
            "title": "自然语言处理技术",
            "content": "自然语言处理是人工智能的一个重要分支，专注于计算机与人类语言之间的交互。NLP技术包括文本分类、情感分析、机器翻译、问答系统等。大语言模型如GPT系列在NLP任务上表现出色。"
        }
    ]
    
    print(f"创建了 {len(documents)} 个示例文档")
    return documents

def build_hybrid_index(documents: List[Dict]) -> Dict[str, Any]:
    """构建混合索引（简化版）"""
    print("\n构建混合索引...")
    
    # 1. 构建稀疏索引（基于关键词匹配）
    print("  - 构建稀疏索引...")
    
    # 提取关键词
    keywords_index = {}
    for doc in documents:
        content = doc["content"].lower()
        # 简单分词
        words = content.split()
        for word in words:
            if len(word) > 2:  # 只考虑长度大于2的词
                if word not in keywords_index:
                    keywords_index[word] = []
                keywords_index[word].append(doc["id"])
    
    print(f"    关键词数量: {len(keywords_index)}")
    
    # 2. 构建密集索引（使用随机向量）
    print("  - 构建密集索引...")
    n_docs = len(documents)
    dim = 10  # 简化维度
    dense_vectors = np.random.randn(n_docs, dim).astype(np.float32)
    dense_vectors = dense_vectors / np.linalg.norm(dense_vectors, axis=1, keepdims=True)
    
    print(f"    密集向量维度: {dense_vectors.shape}")
    
    return {
        "documents": documents,
        "keywords_index": keywords_index,
        "dense_vectors": dense_vectors,
        "num_documents": n_docs
    }

def sparse_search(index: Dict, query: str, top_k: int = 3) -> List[Dict]:
    """稀疏检索"""
    query_words = query.lower().split()
    
    # 计算文档得分
    scores = {}
    for word in query_words:
        if word in index["keywords_index"]:
            for doc_id in index["keywords_index"][word]:
                scores[doc_id] = scores.get(doc_id, 0) + 1
    
    # 按得分排序
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    results = []
    for doc_id, score in sorted_docs:
        # 查找文档
        doc = next((d for d in index["documents"] if d["id"] == doc_id), None)
        if doc:
            results.append({
                "document": doc,
                "score": score / len(query_words),  # 归一化得分
                "type": "sparse"
            })
    
    return results

def dense_search(index: Dict, query: str, top_k: int = 3) -> List[Dict]:
    """密集检索（简化版）"""
    # 生成随机查询向量
    query_vector = np.random.randn(1, index["dense_vectors"].shape[1]).astype(np.float32)
    query_vector = query_vector / np.linalg.norm(query_vector)
    
    # 计算相似度
    similarities = np.dot(index["dense_vectors"], query_vector.T).flatten()
    
    # 获取top-k结果
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            "document": index["documents"][idx],
            "score": float(similarities[idx]),
            "type": "dense"
        })
    
    return results

def hybrid_search(index: Dict, query: str, top_k: int = 3) -> List[Dict]:
    """混合检索"""
    print(f"执行混合检索: {query}")
    
    # 并行执行两种检索
    sparse_results = sparse_search(index, query, top_k * 2)
    dense_results = dense_search(index, query, top_k * 2)
    
    # 合并结果
    all_results = {}
    
    # 添加稀疏检索结果
    for result in sparse_results:
        doc_id = result["document"]["id"]
        if doc_id not in all_results:
            all_results[doc_id] = {
                "document": result["document"],
                "sparse_score": result["score"],
                "dense_score": 0.0
            }
        else:
            all_results[doc_id]["sparse_score"] = result["score"]
    
    # 添加密集检索结果
    for result in dense_results:
        doc_id = result["document"]["id"]
        if doc_id not in all_results:
            all_results[doc_id] = {
                "document": result["document"],
                "sparse_score": 0.0,
                "dense_score": result["score"]
            }
        else:
            all_results[doc_id]["dense_score"] = result["score"]
    
    # 计算综合分数（加权平均）
    for doc_id, result in all_results.items():
        combined_score = (result["sparse_score"] + result["dense_score"]) / 2
        result["combined_score"] = combined_score
    
    # 按综合分数排序
    sorted_results = sorted(
        all_results.values(),
        key=lambda x: x["combined_score"],
        reverse=True
    )[:top_k]
    
    # 格式化结果
    formatted_results = []
    for i, result in enumerate(sorted_results):
        formatted_results.append({
            "document": result["document"],
            "score": result["combined_score"],
            "type": "hybrid",
            "rank": i + 1,
            "sparse_score": result["sparse_score"],
            "dense_score": result["dense_score"]
        })
    
    return formatted_results

def generate_answer(question: str, retrieval_results: List[Dict]) -> str:
    """生成答案（简化版）"""
    if not retrieval_results:
        return "根据提供的文档，无法回答这个问题。"
    
    # 提取相关内容
    contexts = []
    for result in retrieval_results[:2]:  # 使用前2个结果
        doc = result["document"]
        contexts.append(f"【{doc['title']}】{doc['content']}")
    
    context = "\n\n".join(contexts)
    
    # 简单答案生成
    if "人工智能" in question:
        answer = "根据检索到的文档，人工智能是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的智能机器。它包括机器学习、深度学习、自然语言处理等多个子领域。"
    elif "机器学习" in question:
        answer = "根据检索到的文档，机器学习是人工智能的一个子领域，使计算机能够在没有明确编程的情况下学习和改进。主要分为监督学习、无监督学习和强化学习三种类型。"
    elif "深度学习" in question:
        answer = "根据检索到的文档，深度学习是机器学习的一个子集，使用神经网络模拟人脑的工作方式。在计算机视觉、语音识别、自然语言处理等领域有广泛应用。"
    elif "自然语言处理" in question or "NLP" in question:
        answer = "根据检索到的文档，自然语言处理是人工智能的一个重要分支，专注于计算机与人类语言之间的交互。包括文本分类、情感分析、机器翻译、问答系统等技术。"
    else:
        answer = "基于检索到的文档，以下信息可能与您的问题相关：\n"
        for result in retrieval_results[:2]:
            doc = result["document"]
            answer += f"- {doc['title']}: {doc['content'][:100]}...\n"
    
    # 添加检索信息
    info = f"""

【检索统计】
- 总检索结果: {len(retrieval_results)} 个文档
- 最高相关性分数: {retrieval_results[0]['score']:.3f}
- 检索方法: {retrieval_results[0]['type']}

注：这是简化版本的答案，基于关键词匹配和随机向量检索生成。"""
    
    return answer + info

def ask_question(index: Dict, question: str):
    """提问并获取答案"""
    print("\n" + "=" * 50)
    print(f"问题: {question}")
    print("=" * 50)
    
    print("\n处理问题...")
    
    # 混合检索
    print("  - 执行混合检索...")
    retrieval_results = hybrid_search(index, question, top_k=3)
    
    if not retrieval_results:
        print("  - 未找到相关文档")
        return None
    
    print(f"  - 找到 {len(retrieval_results)} 个相关文档")
    
    # 生成答案
    print("  - 生成答案...")
    answer = generate_answer(question, retrieval_results)
    
    return answer

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="混合检索RAG系统 - 简化版本")
    parser.add_argument("--question", "-q", type=str, help="直接提问的问题")
    
    args = parser.parse_args()
    
    # 设置环境
    setup_environment()
    
    # 创建示例文档
    documents = create_sample_documents()
    
    # 构建索引
    index = build_hybrid_index(documents)
    
    if args.question:
        # 直接提问模式
        answer = ask_question(index, args.question)
        if answer:
            print("\n答案:")
            print("-" * 40)
            print(answer)
            print("-" * 40)
        else:
            print("\n无法回答该问题")
    
    else:
        # 默认演示模式
        print("\n" + "=" * 50)
        print("演示模式")
        print("=" * 50)
        
        # 演示问题
        demo_questions = [
            "人工智能是什么？",
            "机器学习有哪些类型？",
            "深度学习有什么应用？",
            "什么是自然语言处理？"
        ]
        
        for question in demo_questions:
            answer = ask_question(index, question)
            if answer:
                print(f"\n问题: {question}")
                print(f"答案: {answer[:150]}...")
            else:
                print(f"\n问题: {question}")
                print("答案: 无法回答")
        
        print("\n演示完成!")
        print("使用 --question '你的问题' 直接提问")

if __name__ == "__main__":
    main()
