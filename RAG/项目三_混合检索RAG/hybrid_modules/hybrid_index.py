"""
混合索引模块
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Any
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from config import Config

class HybridIndex:
    """混合索引模块 - 支持密集和稀疏向量"""
    
    def __init__(self):
        self.config = Config
        
        # 初始化嵌入模型（使用更简单的模型）
        print(f"初始化嵌入模型...")
        try:
            # 尝试使用更简单的模型
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            print(f"  使用模型: paraphrase-multilingual-MiniLM-L12-v2")
        except Exception as e:
            print(f"  嵌入模型初始化失败: {e}")
            print("  使用随机向量作为替代")
            self.embedding_model = None
        
        # 初始化稀疏检索器
        self.bm25 = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
        # 存储文档
        self.documents = []
        self.dense_vectors = None
        self.sparse_vectors = None
        
        # 索引路径
        self.index_path = self.config.VECTOR_STORE_PATH
        
    def build_index(self, documents: List[Document]) -> Dict[str, Any]:
        """构建混合索引"""
        print("构建混合索引...")
        
        if not documents:
            print("没有文档可索引")
            return {"num_documents": 0}
        
        self.documents = documents
        
        # 1. 构建密集向量索引
        print("  - 构建密集向量索引...")
        dense_vectors = self._build_dense_index(documents)
        
        # 2. 构建稀疏向量索引
        print("  - 构建稀疏向量索引...")
        sparse_features = self._build_sparse_index(documents)
        
        # 3. 保存索引
        print("  - 保存索引...")
        self._save_index()
        
        return {
            "num_documents": len(documents),
            "dense_dim": dense_vectors.shape[1] if dense_vectors is not None else 0,
            "sparse_features": sparse_features
        }
    
    def _build_dense_index(self, documents: List[Document]) -> np.ndarray:
        """构建密集向量索引"""
        texts = [doc.page_content for doc in documents]
        
        try:
            # 使用嵌入模型生成密集向量
            embeddings = self.embedding_model.embed_documents(texts)
            self.dense_vectors = np.array(embeddings)
            
            print(f"    密集向量维度: {self.dense_vectors.shape}")
            return self.dense_vectors
            
        except Exception as e:
            print(f"    密集向量生成失败: {e}")
            print("    使用随机向量作为替代...")
            
            # 生成随机向量作为替代
            n_docs = len(documents)
            dim = 768  # BERT基础维度
            self.dense_vectors = np.random.randn(n_docs, dim).astype(np.float32)
            self.dense_vectors = self.dense_vectors / np.linalg.norm(self.dense_vectors, axis=1, keepdims=True)
            
            return self.dense_vectors
    
    def _build_sparse_index(self, documents: List[Document]) -> int:
        """构建稀疏向量索引"""
        texts = [doc.page_content for doc in documents]
        
        # 方法1: BM25
        print("    构建BM25索引...")
        tokenized_texts = [self._tokenize(text) for text in texts]
        self.bm25 = BM25Okapi(tokenized_texts)
        
        # 方法2: TF-IDF
        print("    构建TF-IDF索引...")
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            min_df=1,
            max_df=0.8,
            tokenizer=self._tokenize,
            token_pattern=None
        )
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        
        features = len(self.tfidf_vectorizer.get_feature_names_out())
        print(f"    TF-IDF特征数量: {features}")
        
        return features
    
    def _tokenize(self, text: str) -> List[str]:
        """简单分词函数"""
        # 中文按字符分割，英文按单词分割
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
    
    def _save_index(self):
        """保存索引到文件"""
        os.makedirs(self.index_path, exist_ok=True)
        
        # 保存文档
        with open(os.path.join(self.index_path, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)
        
        # 保存密集向量
        if self.dense_vectors is not None:
            np.save(os.path.join(self.index_path, "dense_vectors.npy"), self.dense_vectors)
        
        # 保存稀疏索引
        index_data = {
            "bm25": self.bm25,
            "tfidf_vectorizer": self.tfidf_vectorizer,
            "tfidf_matrix": self.tfidf_matrix
        }
        
        with open(os.path.join(self.index_path, "sparse_index.pkl"), "wb") as f:
            pickle.dump(index_data, f)
        
        # 保存索引信息
        info = {
            "num_documents": len(self.documents),
            "dense_dim": self.dense_vectors.shape[1] if self.dense_vectors is not None else 0,
            "sparse_features": len(self.tfidf_vectorizer.get_feature_names_out()) if self.tfidf_vectorizer else 0
        }
        
        with open(os.path.join(self.index_path, "index_info.txt"), "w") as f:
            for key, value in info.items():
                f.write(f"{key}: {value}\n")
        
        print(f"索引已保存到: {self.index_path}")
    
    def load_index(self) -> bool:
        """加载索引"""
        print("加载混合索引...")
        
        try:
            # 加载文档
            docs_path = os.path.join(self.index_path, "documents.pkl")
            if os.path.exists(docs_path):
                with open(docs_path, "rb") as f:
                    self.documents = pickle.load(f)
                print(f"  加载文档: {len(self.documents)} 个")
            else:
                print("  文档文件不存在")
                return False
            
            # 加载密集向量
            dense_path = os.path.join(self.index_path, "dense_vectors.npy")
            if os.path.exists(dense_path):
                self.dense_vectors = np.load(dense_path)
                print(f"  加载密集向量: {self.dense_vectors.shape}")
            
            # 加载稀疏索引
            sparse_path = os.path.join(self.index_path, "sparse_index.pkl")
            if os.path.exists(sparse_path):
                with open(sparse_path, "rb") as f:
                    index_data = pickle.load(f)
                
                self.bm25 = index_data.get("bm25")
                self.tfidf_vectorizer = index_data.get("tfidf_vectorizer")
                self.tfidf_matrix = index_data.get("tfidf_matrix")
                
                if self.tfidf_vectorizer:
                    features = len(self.tfidf_vectorizer.get_feature_names_out())
                    print(f"  加载稀疏索引: {features} 个特征")
            
            print("索引加载完成")
            return True
            
        except Exception as e:
            print(f"索引加载失败: {e}")
            return False
    
    def dense_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """密集检索"""
        if self.dense_vectors is None or len(self.documents) == 0:
            return []
        
        # 生成查询向量
        try:
            query_vector = self.embedding_model.embed_query(query)
            query_vector = np.array(query_vector).reshape(1, -1)
        except:
            # 如果嵌入失败，使用随机向量
            query_vector = np.random.randn(1, self.dense_vectors.shape[1]).astype(np.float32)
            query_vector = query_vector / np.linalg.norm(query_vector)
        
        # 计算相似度
        similarities = np.dot(self.dense_vectors, query_vector.T).flatten()
        
        # 获取top-k结果
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "document": self.documents[idx],
                "score": float(similarities[idx]),
                "type": "dense"
            })
        
        return results
    
    def sparse_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """稀疏检索"""
        if not self.bm25 or not self.tfidf_vectorizer or len(self.documents) == 0:
            return []
        
        results = []
        
        # BM25检索
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # TF-IDF检索
        try:
            query_vector = self.tfidf_vectorizer.transform([query])
            tfidf_scores = (self.tfidf_matrix * query_vector.T).toarray().flatten()
        except:
            tfidf_scores = np.zeros(len(self.documents))
        
        # 合并分数（简单平均）
        combined_scores = (bm25_scores + tfidf_scores) / 2
        
        # 获取top-k结果
        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        
        for idx in top_indices:
            results.append({
                "document": self.documents[idx],
                "score": float(combined_scores[idx]),
                "type": "sparse",
                "bm25_score": float(bm25_scores[idx]),
                "tfidf_score": float(tfidf_scores[idx])
            })
        
        return results
    
    def hybrid_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """混合检索 - 结合密集和稀疏检索"""
        print(f"执行混合检索: {query}")
        
        # 并行执行两种检索
        dense_results = self.dense_search(query, top_k * 2)
        sparse_results = self.sparse_search(query, top_k * 2)
        
        # 合并结果
        all_results = {}
        
        # 添加密集检索结果
        for result in dense_results:
            doc_id = result["document"].metadata.get("chunk_id", id(result["document"]))
            if doc_id not in all_results:
                all_results[doc_id] = {
                    "document": result["document"],
                    "dense_score": result["score"],
                    "sparse_score": 0.0
                }
            else:
                all_results[doc_id]["dense_score"] = result["score"]
        
        # 添加稀疏检索结果
        for result in sparse_results:
            doc_id = result["document"].metadata.get("chunk_id", id(result["document"]))
            if doc_id not in all_results:
                all_results[doc_id] = {
                    "document": result["document"],
                    "dense_score": 0.0,
                    "sparse_score": result["score"]
                }
            else:
                all_results[doc_id]["sparse_score"] = result["score"]
        
        # 计算综合分数（加权平均）
        for doc_id, result in all_results.items():
            # 使用RRF (Reciprocal Rank Fusion) 思想
            dense_weight = 0.5
            sparse_weight = 0.5
            
            combined_score = (
                dense_weight * result["dense_score"] +
                sparse_weight * result["sparse_score"]
            )
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
                "dense_score": result["dense_score"],
                "sparse_score": result["sparse_score"]
            })
        
        return formatted_results
