"""
混合检索器模块
"""

from typing import List, Dict, Any
from config import Config
from .hybrid_index import HybridIndex

class HybridRetriever:
    """混合检索器 - 封装检索逻辑"""
    
    def __init__(self):
        self.config = Config
        self.index = HybridIndex()
        
        # 加载索引
        if not self.index.load_index():
            print("索引不存在或加载失败，需要重新构建索引")
            self.index_loaded = False
        else:
            self.index_loaded = True
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """执行混合检索"""
        if not self.index_loaded:
            print("索引未加载，无法检索")
            return []
        
        if top_k is None:
            top_k = self.config.SPARSE_TOP_K  # 使用配置中的top_k
        
        # 执行混合检索
        results = self.index.hybrid_search(query, top_k)
        
        # 格式化结果
        formatted_results = []
        for result in results:
            doc = result["document"]
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": result["score"],
                "type": result["type"],
                "rank": result["rank"],
                "dense_score": result.get("dense_score", 0.0),
                "sparse_score": result.get("sparse_score", 0.0)
            })
        
        return formatted_results
    
    def dense_retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """仅密集检索"""
        if not self.index_loaded:
            return []
        
        if top_k is None:
            top_k = self.config.DENSE_TOP_K
        
        results = self.index.dense_search(query, top_k)
        
        formatted_results = []
        for result in results:
            doc = result["document"]
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": result["score"],
                "type": "dense"
            })
        
        return formatted_results
    
    def sparse_retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """仅稀疏检索"""
        if not self.index_loaded:
            return []
        
        if top_k is None:
            top_k = self.config.SPARSE_TOP_K
        
        results = self.index.sparse_search(query, top_k)
        
        formatted_results = []
        for result in results:
            doc = result["document"]
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": result["score"],
                "type": "sparse",
                "bm25_score": result.get("bm25_score", 0.0),
                "tfidf_score": result.get("tfidf_score", 0.0)
            })
        
        return formatted_results
    
    def compare_retrieval_methods(self, query: str) -> Dict[str, Any]:
        """比较不同检索方法的结果"""
        if not self.index_loaded:
            return {"error": "索引未加载"}
        
        # 执行三种检索
        hybrid_results = self.retrieve(query, top_k=3)
        dense_results = self.dense_retrieve(query, top_k=3)
        sparse_results = self.sparse_retrieve(query, top_k=3)
        
        # 分析结果
        analysis = {
            "query": query,
            "hybrid_results_count": len(hybrid_results),
            "dense_results_count": len(dense_results),
            "sparse_results_count": len(sparse_results),
            "hybrid_top_score": hybrid_results[0]["score"] if hybrid_results else 0,
            "dense_top_score": dense_results[0]["score"] if dense_results else 0,
            "sparse_top_score": sparse_results[0]["score"] if sparse_results else 0,
            "overlap_hybrid_dense": self._calculate_overlap(hybrid_results, dense_results),
            "overlap_hybrid_sparse": self._calculate_overlap(hybrid_results, sparse_results),
            "overlap_dense_sparse": self._calculate_overlap(dense_results, sparse_results)
        }
        
        return {
            "analysis": analysis,
            "hybrid_results": hybrid_results[:3],
            "dense_results": dense_results[:3],
            "sparse_results": sparse_results[:3]
        }
    
    def _calculate_overlap(self, results1: List[Dict], results2: List[Dict]) -> float:
        """计算两个结果集的重叠度"""
        if not results1 or not results2:
            return 0.0
        
        # 提取文档ID
        ids1 = set([r["metadata"].get("chunk_id", id(r)) for r in results1])
        ids2 = set([r["metadata"].get("chunk_id", id(r)) for r in results2])
        
        # 计算Jaccard相似度
        intersection = len(ids1.intersection(ids2))
        union = len(ids1.union(ids2))
        
        return intersection / union if union > 0 else 0.0
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """获取检索统计信息"""
        if not self.index_loaded:
            return {"error": "索引未加载"}
        
        return {
            "index_loaded": self.index_loaded,
            "num_documents": len(self.index.documents) if self.index.documents else 0,
            "dense_dim": self.index.dense_vectors.shape[1] if self.index.dense_vectors is not None else 0,
            "sparse_features": len(self.index.tfidf_vectorizer.get_feature_names_out()) if self.index.tfidf_vectorizer else 0,
            "config": {
                "sparse_top_k": self.config.SPARSE_TOP_K,
                "dense_top_k": self.config.DENSE_TOP_K,
                "rrf_k": self.config.RRF_K
            }
        }
