import os
from typing import List
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from config import Config

class IndexConstruction:
    """索引构建模块"""
    
    def __init__(self):
        self.config = Config
        self.embeddings = self._load_embeddings()
        self.vector_store = None
        
    def _load_embeddings(self):
        """加载嵌入模型"""
        print(f"正在加载嵌入模型: {self.config.EMBEDDING_MODEL}")
        
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        
        embeddings = HuggingFaceEmbeddings(
            model_name=self.config.EMBEDDING_MODEL,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        
        print("嵌入模型加载完成")
        return embeddings
    
    def build_index(self, documents: List[Document], force_rebuild: bool = False):
        """构建向量索引"""
        print("正在构建向量索引...")
        
        # 检查是否已存在索引文件
        index_file = os.path.join(self.config.VECTOR_STORE_PATH, "index.faiss")
        if not force_rebuild and os.path.exists(index_file):
            print("加载现有向量索引...")
            try:
                self.vector_store = FAISS.load_local(
                    self.config.VECTOR_STORE_PATH,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print(f"索引加载完成，包含 {self.vector_store.index.ntotal} 个向量")
                return self.vector_store
            except Exception as e:
                print(f"加载索引失败: {e}，将创建新索引")
        
        # 构建新索引
        print("创建新的向量索引...")
        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        
        # 保存索引
        self.vector_store.save_local(self.config.VECTOR_STORE_PATH)
        print(f"索引构建完成，包含 {self.vector_store.index.ntotal} 个向量")
        print(f"索引已保存到: {self.config.VECTOR_STORE_PATH}")
        
        return self.vector_store
    
    def load_index(self):
        """加载现有索引"""
        if not os.path.exists(self.config.VECTOR_STORE_PATH):
            raise FileNotFoundError(f"向量索引目录不存在: {self.config.VECTOR_STORE_PATH}")
        
        print("加载向量索引...")
        self.vector_store = FAISS.load_local(
            self.config.VECTOR_STORE_PATH,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"索引加载完成，包含 {self.vector_store.index.ntotal} 个向量")
        return self.vector_store
    
    def search_similar(self, query: str, k: int = None) -> List[Document]:
        """搜索相似文档"""
        if self.vector_store is None:
            self.load_index()
        
        if k is None:
            k = self.config.RETRIEVAL_TOP_K
        
        print(f"搜索查询: {query}")
        results = self.vector_store.similarity_search(query, k=k)
        
        print(f"找到 {len(results)} 个相关文档")
        for i, doc in enumerate(results):
            print(f"  [{i+1}] {doc.page_content[:100]}...")
            print(f"      来源: {doc.metadata.get('source', '未知')}")
        
        return results
