"""
多模态检索器
"""

import os
import numpy as np
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from config import Config

class MultimodalEmbeddings(Embeddings):
    """多模态嵌入模型（简化版）"""
    
    def __init__(self):
        self.config = Config
        print(f"初始化多模态嵌入模型: {self.config.MULTIMODAL_EMBEDDING_MODEL}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """为文档生成嵌入"""
        print(f"为 {len(texts)} 个文档生成嵌入...")
        
        embeddings = []
        for text in texts:
            # 简化版：生成随机嵌入（实际应用中应使用真正的多模态模型）
            embedding = self._generate_embedding(text)
            embeddings.append(embedding)
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """为查询生成嵌入"""
        return self._generate_embedding(text)
    
    def _generate_embedding(self, text: str) -> List[float]:
        """生成嵌入向量（简化版）"""
        # 在实际应用中，这里应该调用真正的多模态嵌入模型
        # 如CLIP、BLIP等
        
        # 简化版：基于文本长度和内容生成伪随机向量
        seed = sum(ord(c) for c in text[:100])  # 使用文本前100个字符的ASCII和作为种子
        np.random.seed(seed % 10000)
        
        # 生成384维向量（与常见嵌入模型维度一致）
        embedding = np.random.randn(384).tolist()
        
        # 归一化
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = [x / norm for x in embedding]
        
        return embedding

class MultimodalRetriever:
    """多模态检索器 - 支持文本和图像检索"""
    
    def __init__(self):
        self.config = Config
        self.embeddings = MultimodalEmbeddings()
        self.vector_store = None
        self.documents = []
        
        print("初始化多模态检索器...")
    
    def build_vector_store(self, documents: List[Document]):
        """构建多模态向量存储"""
        print("构建多模态向量存储...")
        
        if not documents:
            print("❌ 没有文档可用于构建向量存储")
            return False
        
        self.documents = documents
        
        try:
            # 提取文档内容
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            # 构建FAISS向量存储
            print(f"构建FAISS向量存储，文档数: {len(texts)}")
            self.vector_store = FAISS.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas
            )
            
            # 保存向量存储
            self.vector_store.save_local(self.config.VECTOR_STORE_PATH)
            print(f"向量存储已保存到: {self.config.VECTOR_STORE_PATH}")
            
            return True
            
        except Exception as e:
            print(f"❌ 构建向量存储失败: {e}")
            return False
    
    def load_vector_store(self) -> bool:
        """加载向量存储"""
        try:
            if os.path.exists(self.config.VECTOR_STORE_PATH):
                print("加载向量存储...")
                self.vector_store = FAISS.load_local(
                    self.config.VECTOR_STORE_PATH,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                return True
            else:
                print("向量存储不存在")
                return False
        except Exception as e:
            print(f"加载向量存储失败: {e}")
            return False
    
    def multimodal_retrieval(self, query: str, image_path: Optional[str] = None) -> Dict[str, Any]:
        """多模态检索"""
        print("执行多模态检索...")
        
        if not self.vector_store:
            print("❌ 向量存储未初始化")
            return {"success": False, "error": "向量存储未初始化"}
        
        # 构建多模态查询
        multimodal_query = self._build_multimodal_query(query, image_path)
        
        # 执行检索
        try:
            # 相似度检索
            docs_with_scores = self.vector_store.similarity_search_with_score(
                multimodal_query, 
                k=self.config.RETRIEVAL_TOP_K
            )
            
            # 处理检索结果
            retrieved_docs = []
            for doc, score in docs_with_scores:
                retrieved_docs.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score),
                    "modality": doc.metadata.get("modality", "text")
                })
            
            # 分析检索结果
            stats = self._analyze_retrieval_results(retrieved_docs, query, image_path)
            
            return {
                "success": True,
                "retrieved_docs": retrieved_docs,
                "final_docs": len(retrieved_docs),
                "stats": stats,
                "query_type": "multimodal" if image_path else "text_only"
            }
            
        except Exception as e:
            print(f"❌ 检索失败: {e}")
            return {"success": False, "error": str(e)}
    
    def _build_multimodal_query(self, query: str, image_path: Optional[str] = None) -> str:
        """构建多模态查询"""
        if image_path:
            # 如果有图像路径，构建多模态查询
            image_info = f"[图像查询: {os.path.basename(image_path)}] "
            return f"{image_info}{query}"
        else:
            # 纯文本查询
            return query
    
    def _analyze_retrieval_results(self, docs: List[Dict], query: str, image_path: Optional[str] = None) -> Dict[str, Any]:
        """分析检索结果"""
        if not docs:
            return {
                "total_docs": 0,
                "avg_score": 0,
                "modality_distribution": {},
                "query_relevance": 0
            }
        
        # 计算平均分数
        scores = [doc["score"] for doc in docs]
        avg_score = sum(scores) / len(scores)
        
        # 分析模态分布
        modality_count = {}
        for doc in docs:
            modality = doc.get("modality", "unknown")
            modality_count[modality] = modality_count.get(modality, 0) + 1
        
        # 计算查询相关性（简化版）
        query_relevance = self._calculate_query_relevance(docs, query)
        
        return {
            "total_docs": len(docs),
            "avg_score": avg_score,
            "modality_distribution": modality_count,
            "query_relevance": query_relevance,
            "has_image_query": bool(image_path)
        }
    
    def _calculate_query_relevance(self, docs: List[Dict], query: str) -> float:
        """计算查询相关性（简化版）"""
        if not docs:
            return 0.0
        
        # 简单的关键词匹配
        query_words = set(query.lower().split())
        total_relevance = 0.0
        
        for doc in docs:
            content = doc["content"].lower()
            matched_words = sum(1 for word in query_words if word in content)
            relevance = matched_words / max(len(query_words), 1)
            total_relevance += relevance
        
        return total_relevance / len(docs)
    
    def retrieve_pipeline(self, query: str, image_path: Optional[str] = None) -> List[Document]:
        """检索管道"""
        print("执行检索管道...")
        
        # 执行多模态检索
        retrieval_result = self.multimodal_retrieval(query, image_path)
        
        if not retrieval_result.get("success", False):
            print("❌ 检索失败")
            return []
        
        # 转换检索结果为Document对象
        retrieved_docs = retrieval_result.get("retrieved_docs", [])
        
        documents = []
        for doc_info in retrieved_docs:
            doc = Document(
                page_content=doc_info["content"],
                metadata=doc_info["metadata"]
            )
            documents.append(doc)
        
        print(f"检索管道完成: {len(documents)} 个文档")
        return documents
    
    def cross_modal_retrieval_demo(self):
        """跨模态检索演示"""
        print("\n" + "=" * 50)
        print("跨模态检索演示")
        print("=" * 50)
        
        demo_cases = [
            {
                "query": "人工智能图像识别",
                "description": "文本到图像检索示例",
                "expected_modality": "image"
            },
            {
                "query": "多模态学习技术",
                "description": "文本到文本检索示例",
                "expected_modality": "text"
            },
            {
                "query": "CLIP模型原理",
                "description": "跨模态模型检索示例",
                "expected_modality": "both"
            }
        ]
        
        for i, case in enumerate(demo_cases, 1):
            print(f"\n案例 {i}: {case['description']}")
            print(f"查询: {case['query']}")
            
            result = self.multimodal_retrieval(case["query"])
            
            if result.get("success", False):
                docs = result.get("retrieved_docs", [])
                print(f"检索到 {len(docs)} 个文档")
                
                # 显示模态分布
                stats = result.get("stats", {})
                modality_dist = stats.get("modality_distribution", {})
                print(f"模态分布: {modality_dist}")
                
                # 显示前2个结果
                for j, doc in enumerate(docs[:2], 1):
                    modality = doc.get("modality", "unknown")
                    score = doc.get("score", 0)
                    print(f"  结果{j}: [{modality}] 分数: {score:.3f}")
            else:
                print("❌ 检索失败")
            
            print("-" * 40)
