import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class Config:
    """项目配置类"""
    
    # Neo4j配置
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    
    # DeepSeek配置
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "yourskindeepseek")
    DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
    DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    
    # 嵌入模型配置 - 使用更简单的模型避免下载问题
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    
    # 数据路径
    DATA_PATH = "data/raw_texts"
    
    # 分块配置
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100
    
    # 检索配置
    VECTOR_TOP_K = 3
    GRAPH_TOP_K = 3
    
    # 生成配置
    MAX_TOKENS = 1000
    TEMPERATURE = 0.7
    
    @classmethod
    def validate(cls):
        """验证配置"""
        if not cls.DEEPSEEK_API_KEY:
            raise ValueError("请设置DEEPSEEK_API_KEY环境变量")
        
        # 创建必要的目录
        os.makedirs(cls.DATA_PATH, exist_ok=True)
        
        print("配置验证通过")
