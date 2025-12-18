import os
from typing import List, Dict, Any
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from config import Config

class RetrievalOptimization:
    """检索优化模块"""
    
    def __init__(self):
        self.config = Config
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vector_store = None
    
    def create_vector_store(self, chunks: List[Document]) -> FAISS:
        """创建向量存储"""
        print("正在创建向量存储...")
        
        # 创建FAISS向量存储
        self.vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        
        print(f"向量存储创建完成，包含 {len(chunks)} 个文档")
        return self.vector_store
    
    def load_vector_store(self) -> FAISS:
        """加载向量存储"""
        print("正在加载向量存储...")
        
        if os.path.exists(self.config.VECTOR_STORE_PATH):
            self.vector_store = FAISS.load_local(
                self.config.VECTOR_STORE_PATH,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"向量存储加载完成，包含 {self.vector_store.index.ntotal} 个向量")
        else:
            print("向量存储不存在，请先创建")
            self.vector_store = None
        
        return self.vector_store
    
    def save_vector_store(self):
        """保存向量存储"""
        if self.vector_store:
            self.vector_store.save_local(self.config.VECTOR_STORE_PATH)
            print(f"向量存储已保存到 {self.config.VECTOR_STORE_PATH}")
    
    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        """检索相关文档"""
        if not self.vector_store:
            print("向量存储未初始化，请先创建或加载")
            return []
        
        print(f"正在检索查询: {query}")
        results = self.vector_store.similarity_search(query, k=k)
        
        print(f"检索到 {len(results)} 个相关文档:")
        for i, doc in enumerate(results):
            source = doc.metadata.get("source", "未知")
            print(f"  {i+1}. {source} (相似度: {doc.metadata.get('score', 'N/A')})")
        
        return results
    
    def optimize_retrieval(self, query: str, chunks: List[Document] = None) -> List[Document]:
        """优化检索流程"""
        # 如果向量存储不存在，先创建
        if not self.vector_store:
            if chunks:
                self.create_vector_store(chunks)
            else:
                print("无法创建向量存储：没有提供文档块")
                return []
        
        # 执行检索
        results = self.retrieve(query)
        return results
