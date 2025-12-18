#!/usr/bin/env python3
"""
基于知识图谱的RAG系统主程序
"""

import sys
import argparse
from config import Config
from kg_modules import KnowledgeExtraction, GraphConstruction, GraphRetrieval, HybridRAG

def setup_environment():
    """设置环境"""
    print("=" * 50)
    print("基于知识图谱的RAG系统")
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
    """构建知识图谱RAG系统"""
    print("\n" + "=" * 50)
    print("构建知识图谱RAG系统")
    print("=" * 50)
    
    # 知识抽取
    print("\n1. 知识抽取...")
    knowledge_extractor = KnowledgeExtraction()
    knowledge_data = knowledge_extractor.prepare_knowledge_data()
    
    # 图构建（简化版，实际应该使用Neo4j）
    print("\n2. 图构建...")
    graph_builder = GraphConstruction()
    graph_data = graph_builder.build_graph(knowledge_data)
    
    # 向量索引构建 - 改进的简化版本
    print("\n3. 向量索引构建（改进版）...")
    
    # 创建一个改进的向量检索器
    class ImprovedVectorStore:
        """改进的向量存储，使用更好的匹配算法"""
        def __init__(self, documents):
            self.documents = documents
            self.keyword_index = self._build_keyword_index(documents)
            print(f"创建了包含 {len(documents)} 个文档的改进向量存储")
        
        def _build_keyword_index(self, documents):
            """构建关键词索引"""
            index = {}
            for i, doc in enumerate(documents):
                content = doc.page_content.lower()
                # 提取关键词（中文词汇）
                import re
                words = re.findall(r'[\u4e00-\u9fff]{2,6}', content)
                for word in words:
                    if word not in index:
                        index[word] = []
                    index[word].append(i)
            return index
        
        def similarity_search(self, query: str, k: int = 3):
            """改进的相似度搜索 - 基于关键词索引"""
            print(f"执行改进向量搜索: {query}")
            
            # 提取查询中的中文词汇
            import re
            query_words = re.findall(r'[\u4e00-\u9fff]{2,6}', query.lower())
            
            if not query_words:
                # 如果没有中文词汇，使用原始查询词
                query_words = query.lower().split()
            
            # 计算文档得分
            doc_scores = {}
            for word in query_words:
                if word in self.keyword_index:
                    for doc_idx in self.keyword_index[word]:
                        if doc_idx not in doc_scores:
                            doc_scores[doc_idx] = 0
                        doc_scores[doc_idx] += 1
            
            # 按得分排序
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            
            # 获取前k个文档
            results = []
            for doc_idx, score in sorted_docs[:k]:
                results.append(self.documents[doc_idx])
            
            # 如果没有找到，返回所有文档的前几个
            if not results and self.documents:
                results = self.documents[:min(k, len(self.documents))]
            
            print(f"找到 {len(results)} 个相关文档")
            return results
    
    # 创建改进向量存储
    vector_store = ImprovedVectorStore(knowledge_data["documents"])
    vector_store_path = "vector_index"
    
    # 创建目录（如果需要）
    import os
    os.makedirs(vector_store_path, exist_ok=True)
    
    # 保存索引信息（简化版）
    with open(os.path.join(vector_store_path, "index_info.txt"), "w") as f:
        f.write(f"文档数量: {len(knowledge_data['documents'])}\n")
        f.write(f"实体数量: {len(knowledge_data.get('entities', []))}\n")
        f.write(f"关系数量: {len(knowledge_data.get('relations', []))}\n")
    
    print(f"向量索引已保存到: {vector_store_path}")
    
    # 创建检索器
    def vector_retriever(query: str, k: int = Config.VECTOR_TOP_K):
        results = vector_store.similarity_search(query, k=k)
        return results
    
    # 图检索器（简化版）
    def graph_retriever(query: str, k: int = Config.GRAPH_TOP_K):
        graph_retriever_obj = GraphRetrieval(graph_data)
        results = graph_retriever_obj.retrieve_from_graph(query, k=k)
        return results
    
    # 混合RAG
    print("\n4. 创建混合RAG系统...")
    hybrid_rag = HybridRAG(vector_retriever, graph_retriever)
    
    return hybrid_rag

def interactive_mode(hybrid_rag):
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
            print("\n处理中...")
            answer = hybrid_rag.answer_question(question)
            
            print("\n" + "-" * 50)
            print("答案:")
            print(answer)
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n\n用户中断，退出系统")
            break
        except Exception as e:
            print(f"错误: {e}")

def single_question_mode(hybrid_rag, question):
    """单个问题模式"""
    print("\n" + "=" * 50)
    print(f"问题: {question}")
    print("=" * 50)
    
    answer = hybrid_rag.answer_question(question)
    
    print("\n答案:")
    print(answer)
    print("=" * 50)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="基于知识图谱的RAG系统")
    parser.add_argument("--question", "-q", type=str, help="单个问题")
    parser.add_argument("--rebuild", "-r", action="store_true", help="重新构建系统")
    
    args = parser.parse_args()
    
    # 设置环境
    setup_environment()
    
    # 构建系统
    hybrid_rag = build_system()
    
    # 根据参数选择模式
    if args.question:
        # 单个问题模式
        single_question_mode(hybrid_rag, args.question)
    else:
        # 交互模式
        interactive_mode(hybrid_rag)

if __name__ == "__main__":
    main()
