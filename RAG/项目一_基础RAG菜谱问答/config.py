import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class Config:
    """项目配置类"""
    
    # DeepSeek配置
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "yourskindeepseek")
    DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
    DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    
    # 嵌入模型配置
    EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
    
    # 向量数据库配置
    VECTOR_STORE_PATH = "vector_index"
    
    # 数据路径
    DATA_PATH = "data/recipes"
    
    # 分块配置
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100
    
    # 检索配置
    RETRIEVAL_TOP_K = 3
    
    # 生成配置
    MAX_TOKENS = 1000
    TEMPERATURE = 0.7
    
    @classmethod
    def validate(cls):
        """验证配置"""
        if not cls.DEEPSEEK_API_KEY:
            raise ValueError("请设置DEEPSEEK_API_KEY环境变量")
        
        # 创建必要的目录
        os.makedirs(cls.VECTOR_STORE_PATH, exist_ok=True)
        os.makedirs(cls.DATA_PATH, exist_ok=True)
        
        print("配置验证通过")
