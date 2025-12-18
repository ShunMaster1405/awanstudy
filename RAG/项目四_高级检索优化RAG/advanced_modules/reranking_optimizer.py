"""
重排序优化器模块
"""

import numpy as np
from typing import List, Dict, Any
from langchain.schema import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from config import Config

class RerankingOptimizer:
    """重排序优化器 - 支持多种重排序方法"""
    
    def __init__(self):
        self.config = Config
        
        # 初始化TF-IDF向量化器
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            min_df=1,
            max_df=0.8,
            tokenizer=self._tokenize,
            token_pattern=None
        )
        
        # BM25索引
        self.bm25 = None
        self.tfidf_matrix = None
        self.documents = []
    
    def _tokenize(self, text: str) -> List[str]:
        """简单分词函数"""
        tokens = []
        current_word = ""
        
        for char in text:
            if '\u4e00' <= char <= '\u9fff':  # 中文字符
                if current_word:
                    tokens.append(current_word.lower())
                    current_word = ""
                tokens.append(char)
            elif char.isalpha():  # 英文字母
                current_word += char
            else:  # 其他字符
                if current_word:
                    tokens.append(current_word.lower())
                    current_word = ""
        
        if current_word:
            tokens.append(current_word.lower())
        
        return tokens
    
    def build_index(self, documents: List[Document]):
        """构建重排序索引"""
        print("构建重排序索引...")
        
        self.documents = documents
        texts = [doc.page_content for doc in documents]
        
        # 构建BM25索引
        print("  - 构建BM25索引...")
        tokenized_texts = [self._tokenize(text) for text in texts]
        self.bm25 = BM25Okapi(tokenized_texts)
        
        # 构建TF-IDF矩阵
        print("  - 构建TF-IDF矩阵...")
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        
        features = len(self.tfidf_vectorizer.get_feature_names_out())
        print(f"  - TF-IDF特征数量: {features}")
    
    def bm25_rerank(self, query: str, documents: List[Document], top_k: int = None) -> List[Dict[str, Any]]:
        """BM25重排序"""
        if not self.bm25:
            print("BM25索引未构建")
            return []
        
        texts = [doc.page_content for doc in documents]
        tokenized_query = self._tokenize(query)
        
        # 计算BM25分数
        scores = self.bm25.get_scores(tokenized_query)
        
        # 创建结果列表
        results = []
        for i, (doc, score) in enumerate(zip(documents, scores)):
            results.append({
                "document": doc,
                "score": float(score),
                "method": "bm25",
                "rank": i + 1
            })
        
        # 按分数排序
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # 限制返回数量
        if top_k:
            results = results[:top_k]
        
        return results
    
    def tfidf_rerank(self, query: str, documents: List[Document], top_k: int = None) -> List[Dict[str, Any]]:
        """TF-IDF重排序"""
        if not self.tfidf_vectorizer or self.tfidf_matrix is None:
            print("TF-IDF索引未构建")
            return []
        
        texts = [doc.page_content for doc in documents]
        
        try:
            # 转换查询向量
            query_vector = self.tfidf_vectorizer.transform([query])
            
            # 计算TF-IDF相似度
            similarities = (self.tfidf_matrix * query_vector.T).toarray().flatten()
            
            # 创建结果列表
            results = []
            for i, (doc, score) in enumerate(zip(documents, similarities)):
                results.append({
                    "document": doc,
                    "score": float(score),
                    "method": "tfidf",
                    "rank": i + 1
                })
            
            # 按分数排序
            results.sort(key=lambda x: x["score"], reverse=True)
            
            # 限制返回数量
            if top_k:
                results = results[:top_k]
            
            return results
            
        except Exception as e:
            print(f"TF-IDF重排序失败: {e}")
            return []
    
    def hybrid_rerank(self, query: str, documents: List[Document], top_k: int = None) -> List[Dict[str, Any]]:
        """混合重排序（BM25 + TF-IDF）"""
        print(f"执行混合重排序: {query}")
        
        # 分别执行两种重排序
        bm25_results = self.bm25_rerank(query, documents, top_k * 2 if top_k else None)
        tfidf_results = self.tfidf_rerank(query, documents, top_k * 2 if top_k else None)
        
        # 如果两种方法都失败，返回空列表
        if not bm25_results and not tfidf_results:
            print("BM25和TF-IDF重排序都失败")
            return []
        
        # 合并结果
        all_results = {}
        
        # 添加BM25结果
        for result in bm25_results:
            doc_id = result["document"].metadata.get("chunk_id", id(result["document"]))
            if doc_id not in all_results:
                all_results[doc_id] = {
                    "document": result["document"],
                    "bm25_score": result["score"],
                    "tfidf_score": 0.0
                }
            else:
                all_results[doc_id]["bm25_score"] = result["score"]
        
        # 添加TF-IDF结果
        for result in tfidf_results:
            doc_id = result["document"].metadata.get("chunk_id", id(result["document"]))
            if doc_id not in all_results:
                all_results[doc_id] = {
                    "document": result["document"],
                    "bm25_score": 0.0,
                    "tfidf_score": result["score"]
                }
            else:
                all_results[doc_id]["tfidf_score"] = result["score"]
        
        # 计算综合分数（加权平均）
        for doc_id, result in all_results.items():
            # 归一化分数
            bm25_max = max([r["score"] for r in bm25_results], default=1.0)
            tfidf_max = max([r["score"] for r in tfidf_results], default=1.0)
            
            bm25_norm = result["bm25_score"] / max(bm25_max, 0.001)
            tfidf_norm = result["tfidf_score"] / max(tfidf_max, 0.001)
            
            # 加权平均
            combined_score = 0.6 * bm25_norm + 0.4 * tfidf_norm
            result["combined_score"] = combined_score
        
        # 按综合分数排序
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x["combined_score"],
            reverse=True
        )
        
        # 格式化结果
        formatted_results = []
        for i, result in enumerate(sorted_results):
            formatted_results.append({
                "document": result["document"],
                "score": result["combined_score"],
                "method": "hybrid",
                "rank": i + 1,
                "bm25_score": result["bm25_score"],
                "tfidf_score": result["tfidf_score"]
            })
        
        # 限制返回数量
        if top_k:
            formatted_results = formatted_results[:top_k]
        
        return formatted_results
    
    def reciprocal_rank_fusion(self, results_list: List[List[Dict[str, Any]]], k: int = 60) -> List[Dict[str, Any]]:
        """互惠排名融合（RRF）"""
        print("执行互惠排名融合（RRF）...")
        
        # 收集所有文档
        all_docs = {}
        
        for rank_list in results_list:
            for rank, result in enumerate(rank_list, 1):
                doc_id = result["document"].metadata.get("chunk_id", id(result["document"]))
                
                if doc_id not in all_docs:
                    all_docs[doc_id] = {
                        "document": result["document"],
                        "scores": [],
                        "ranks": []
                    }
                
                all_docs[doc_id]["scores"].append(result["score"])
                all_docs[doc_id]["ranks"].append(rank)
        
        # 计算RRF分数
        for doc_id, doc_info in all_docs.items():
            rrf_score = 0.0
            for rank in doc_info["ranks"]:
                rrf_score += 1.0 / (k + rank)
            doc_info["rrf_score"] = rrf_score
        
        # 按RRF分数排序
        sorted_docs = sorted(
            all_docs.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )
        
        # 格式化结果
        formatted_results = []
        for i, doc_info in enumerate(sorted_docs):
            formatted_results.append({
                "document": doc_info["document"],
                "score": doc_info["rrf_score"],
                "method": "rrf",
                "rank": i + 1,
                "original_scores": doc_info["scores"],
                "original_ranks": doc_info["ranks"]
            })
        
        return formatted_results
    
    def rerank_pipeline(self, query: str, documents: List[Document], top_k: int = None) -> List[Dict[str, Any]]:
        """重排序管道"""
        print("=" * 50)
        print("重排序管道")
        print("=" * 50)
        
        if not documents:
            print("没有文档可重排序")
            return []
        
        # 构建索引（如果需要）
        if not self.bm25 or not self.tfidf_vectorizer:
            self.build_index(documents)
        
        print(f"查询: {query}")
        print(f"输入文档数: {len(documents)}")
        
        # 执行混合重排序
        print("\n1. 执行混合重排序...")
        hybrid_results = self.hybrid_rerank(query, documents, top_k)
        
        if not hybrid_results:
            print("重排序失败，返回原始文档")
            results = []
            for i, doc in enumerate(documents[:top_k] if top_k else documents):
                results.append({
                    "document": doc,
                    "score": 1.0 - (i * 0.1),  # 简单递减分数
                    "method": "fallback",
                    "rank": i + 1
                })
            return results
        
        print(f"重排序完成: {len(hybrid_results)} 个结果")
        
        # 显示前几个结果
        print("\n前5个重排序结果:")
        for i, result in enumerate(hybrid_results[:5]):
            doc = result["document"]
            title = doc.metadata.get("title", "未知标题")
            score = result["score"]
            method = result["method"]
            print(f"  {i+1}. [{method}] {title} (分数: {score:.3f})")
        
        return hybrid_results
