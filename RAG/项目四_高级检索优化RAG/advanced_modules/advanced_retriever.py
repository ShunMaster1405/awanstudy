"""
高级检索器模块 - 集成多种优化技术
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from config import Config
from .reranking_optimizer import RerankingOptimizer
from .compression_optimizer import CompressionOptimizer
from .correction_optimizer import CorrectionOptimizer

class AdvancedRetriever:
    """高级检索器 - 集成重排序、压缩、校正技术"""
    
    def __init__(self):
        self.config = Config
        
        # 初始化组件
        self.reranker = RerankingOptimizer()
        self.compressor = CompressionOptimizer()
        self.corrector = CorrectionOptimizer()
        
        # 向量存储
        self.vector_store = None
        self.embedding_model = None
        
        # 文档缓存
        self.documents = []
    
    def build_vector_store(self, documents: List[Document]):
        """构建向量存储"""
        print("构建向量存储...")
        
        self.documents = documents
        
        # 初始化嵌入模型
        print("初始化嵌入模型...")
        try:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=self.config.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            print(f"  使用模型: {self.config.EMBEDDING_MODEL}")
        except Exception as e:
            print(f"嵌入模型初始化失败: {e}")
            print("使用备用模型...")
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        
        # 构建FAISS向量存储
        print("构建FAISS向量存储...")
        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embedding_model
        )
        
        # 保存向量存储
        vector_store_path = self.config.VECTOR_STORE_PATH
        os.makedirs(vector_store_path, exist_ok=True)
        
        self.vector_store.save_local(vector_store_path)
        print(f"向量存储已保存到: {vector_store_path}")
        
        # 构建重排序索引
        self.reranker.build_index(documents)
        
        print(f"向量存储构建完成: {len(documents)} 个文档")
    
    def load_vector_store(self):
        """加载向量存储"""
        print("加载向量存储...")
        
        vector_store_path = self.config.VECTOR_STORE_PATH
        
        if not os.path.exists(vector_store_path):
            print(f"向量存储路径不存在: {vector_store_path}")
            return False
        
        try:
            # 初始化嵌入模型
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=self.config.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # 加载FAISS向量存储
            self.vector_store = FAISS.load_local(
                vector_store_path,
                self.embedding_model,
                allow_dangerous_deserialization=True
            )
            
            # 加载文档
            documents_file = os.path.join(vector_store_path, "documents.pkl")
            if os.path.exists(documents_file):
                with open(documents_file, 'rb') as f:
                    self.documents = pickle.load(f)
            
            # 构建重排序索引
            if self.documents:
                self.reranker.build_index(self.documents)
            
            print(f"向量存储加载完成: {len(self.documents)} 个文档")
            return True
            
        except Exception as e:
            print(f"加载向量存储失败: {e}")
            return False
    
    def basic_retrieval(self, query: str, top_k: int = None) -> List[Document]:
        """基础检索（向量相似度）"""
        if not self.vector_store:
            print("向量存储未初始化")
            return []
        
        if top_k is None:
            top_k = self.config.RETRIEVAL_TOP_K
        
        print(f"执行基础检索: {query}")
        
        try:
            # 使用向量存储进行相似度搜索
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=top_k * 2  # 获取更多结果用于后续处理
            )
            
            # 提取文档并添加分数
            retrieved_docs = []
            for doc, score in results:
                # 添加相关性分数到元数据
                doc.metadata["relevance_score"] = float(1.0 - score)  # 转换距离为相似度
                doc.metadata["retrieval_method"] = "vector_similarity"
                retrieved_docs.append(doc)
            
            print(f"基础检索完成: {len(retrieved_docs)} 个文档")
            return retrieved_docs
            
        except Exception as e:
            print(f"基础检索失败: {e}")
            return []
    
    def reranking_retrieval(self, query: str, documents: List[Document] = None, top_k: int = None) -> List[Dict[str, Any]]:
        """重排序检索"""
        if top_k is None:
            top_k = self.config.RERANK_TOP_K
        
        # 如果没有提供文档，先进行基础检索
        if not documents:
            documents = self.basic_retrieval(query, top_k * 3)
        
        if not documents:
            print("没有文档可重排序")
            return []
        
        print(f"执行重排序检索: {query}")
        
        # 执行重排序
        reranked_results = self.reranker.rerank_pipeline(query, documents, top_k)
        
        # 更新文档的元数据
        for result in reranked_results:
            doc = result["document"]
            doc.metadata["reranking_score"] = result["score"]
            doc.metadata["reranking_method"] = result["method"]
            doc.metadata["reranking_rank"] = result["rank"]
        
        print(f"重排序检索完成: {len(reranked_results)} 个结果")
        return reranked_results
    
    def compression_retrieval(self, query: str, documents: List[Document] = None) -> List[Document]:
        """压缩检索"""
        # 如果没有提供文档，先进行基础检索
        if not documents:
            documents = self.basic_retrieval(query, self.config.RETRIEVAL_TOP_K)
        
        if not documents:
            print("没有文档可压缩")
            return []
        
        print(f"执行压缩检索: {query}")
        
        # 执行压缩
        compressed_docs = self.compressor.compression_pipeline(documents, query)
        
        print(f"压缩检索完成: {len(compressed_docs)} 个文档")
        return compressed_docs
    
    def correction_retrieval(self, query: str, documents: List[Document] = None) -> Tuple[List[Document], Dict[str, Any]]:
        """校正检索 (C-RAG)"""
        # 如果没有提供文档，先进行基础检索
        if not documents:
            documents = self.basic_retrieval(query, self.config.RETRIEVAL_TOP_K)
        
        if not documents:
            print("没有文档可校正")
            return [], {"error": "没有输入文档"}
        
        print(f"执行校正检索: {query}")
        
        # 执行校正
        corrected_docs, correction_info = self.corrector.correction_pipeline(query, documents)
        
        # 标记校正后的文档
        for doc in corrected_docs:
            if "supplementary" in doc.metadata and doc.metadata["supplementary"]:
                doc.metadata["correction_type"] = "supplementary"
            else:
                doc.metadata["correction_type"] = "optimized"
        
        print(f"校正检索完成: {len(corrected_docs)} 个文档")
        return corrected_docs, correction_info
    
    def multi_stage_retrieval(self, query: str) -> Dict[str, Any]:
        """多级检索管道"""
        print("=" * 50)
        print("多级检索管道")
        print("=" * 50)
        
        print(f"查询: {query}")
        
        # 第一阶段：基础检索
        print("\n第一阶段：基础检索")
        stage1_docs = self.basic_retrieval(query, self.config.RETRIEVAL_TOP_K)
        
        if not stage1_docs:
            print("基础检索无结果")
            return {
                "success": False,
                "error": "基础检索无结果",
                "query": query,
                "stage1_docs": 0,
                "stage2_docs": 0,
                "stage3_docs": 0,
                "final_docs": 0
            }
        
        print(f"基础检索结果: {len(stage1_docs)} 个文档")
        
        # 第二阶段：重排序
        print("\n第二阶段：重排序")
        stage2_results = self.reranking_retrieval(query, stage1_docs, self.config.RERANK_TOP_K)
        stage2_docs = [result["document"] for result in stage2_results]
        
        print(f"重排序结果: {len(stage2_docs)} 个文档")
        
        # 第三阶段：压缩
        print("\n第三阶段：压缩")
        stage3_docs = self.compression_retrieval(query, stage2_docs)
        
        print(f"压缩结果: {len(stage3_docs)} 个文档")
        
        # 第四阶段：校正 (C-RAG)
        print("\n第四阶段：校正 (C-RAG)")
        final_docs, correction_info = self.correction_retrieval(query, stage3_docs)
        
        print(f"校正结果: {len(final_docs)} 个文档")
        
        # 收集统计信息
        stats = {
            "success": True,
            "query": query,
            "stage1_docs": len(stage1_docs),
            "stage2_docs": len(stage2_docs),
            "stage3_docs": len(stage3_docs),
            "final_docs": len(final_docs),
            "correction_info": correction_info,
            "retrieval_pipeline": [
                {"stage": "基础检索", "docs": len(stage1_docs)},
                {"stage": "重排序", "docs": len(stage2_docs)},
                {"stage": "压缩", "docs": len(stage3_docs)},
                {"stage": "校正", "docs": len(final_docs)}
            ]
        }
        
        # 显示最终结果摘要
        print("\n" + "=" * 50)
        print("多级检索完成")
        print("=" * 50)
        
        print(f"查询: {query}")
        print(f"最终文档数: {len(final_docs)}")
        
        if final_docs:
            print("\n最终检索结果:")
            for i, doc in enumerate(final_docs[:5]):  # 显示前5个结果
                title = doc.metadata.get("title", "未知标题")
                source = doc.metadata.get("source", "未知来源")
                method = doc.metadata.get("retrieval_method", "未知方法")
                
                # 检查是否为补充文档
                if doc.metadata.get("supplementary", False):
                    source = "[补充] " + source
                
                print(f"  {i+1}. {title} ({source}) - {method}")
            
            if len(final_docs) > 5:
                print(f"  ... 还有 {len(final_docs) - 5} 个文档")
        
        return stats
    
    def retrieve_pipeline(self, query: str, use_correction: bool = True) -> List[Document]:
        """检索管道（简化版）"""
        print(f"执行检索管道: {query}")
        
        # 基础检索
        base_docs = self.basic_retrieval(query)
        
        if not base_docs:
            print("检索无结果")
            return []
        
        # 重排序
        reranked_results = self.reranking_retrieval(query, base_docs)
        reranked_docs = [result["document"] for result in reranked_results]
        
        # 压缩
        compressed_docs = self.compression_retrieval(query, reranked_docs)
        
        # 校正（可选）
        if use_correction:
            final_docs, _ = self.correction_retrieval(query, compressed_docs)
        else:
            final_docs = compressed_docs
        
        print(f"检索管道完成: {len(final_docs)} 个文档")
        return final_docs
