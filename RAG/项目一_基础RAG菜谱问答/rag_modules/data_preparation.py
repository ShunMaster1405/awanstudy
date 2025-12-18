import os
import re
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.schema import Document
from config import Config

class DataPreparation:
    """数据准备模块"""
    
    def __init__(self):
        self.config = Config
        
    def load_recipes(self) -> List[Document]:
        """加载菜谱数据"""
        print("正在加载菜谱数据...")
        
        # 如果数据目录不存在，创建示例数据
        if not os.path.exists(self.config.DATA_PATH) or not os.listdir(self.config.DATA_PATH):
            self._create_sample_recipes()
        
        # 加载Markdown文件
        loader = DirectoryLoader(
            self.config.DATA_PATH,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'}
        )
        
        documents = loader.load()
        print(f"成功加载 {len(documents)} 个菜谱文件")
        return documents
    
    def _create_sample_recipes(self):
        """创建示例菜谱数据"""
        print("创建示例菜谱数据...")
        
        sample_recipes = {
            "宫保鸡丁.md": """# 宫保鸡丁的做法
宫保鸡丁是一道经典的川菜，以鸡肉丁、花生米和干辣椒为主要原料。
预估烹饪难度：★★★

## 必备原料和工具
* 鸡胸肉 300克
* 花生米 100克
* 干辣椒 10个
* 葱、姜、蒜适量
* 食用油、盐、糖、醋、酱油、料酒

## 计算
* 鸡胸肉 = 300克
* 花生米 = 100克
* 干辣椒 = 10个

## 操作
1. 鸡胸肉切丁，用料酒、盐腌制10分钟
2. 花生米用油炸至金黄酥脆
3. 干辣椒剪成小段，葱姜蒜切末
4. 热锅凉油，下鸡丁滑炒至变色
5. 加入干辣椒、葱姜蒜爆香
6. 加入调味料翻炒均匀
7. 最后加入花生米翻炒均匀即可

## 附加内容
可以根据个人口味调整辣度和酸甜比例。""",
            
            "西红柿炒鸡蛋.md": """# 西红柿炒鸡蛋的做法
西红柿炒蛋是中国家常几乎最常见的一道菜肴，简单易做，营养丰富。
预估烹饪难度：★

## 必备原料和工具
* 西红柿 2个
* 鸡蛋 3个
* 食用油、盐、糖适量
* 葱少许

## 计算
* 西红柿 = 2个（约 360g）
* 鸡蛋 = 3个

## 操作
1. 西红柿洗净切块
2. 鸡蛋打散，加少许盐
3. 热锅凉油，倒入鸡蛋液炒熟盛出
4. 锅中再加少许油，下西红柿翻炒
5. 加入适量盐和糖调味
6. 西红柿炒软后加入炒好的鸡蛋
7. 翻炒均匀即可出锅

## 附加内容
喜欢汤汁多的可以加少许水。"""
        }
        
        os.makedirs(self.config.DATA_PATH, exist_ok=True)
        for filename, content in sample_recipes.items():
            filepath = os.path.join(self.config.DATA_PATH, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
        
        print(f"创建了 {len(sample_recipes)} 个示例菜谱")
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档为文本块"""
        print("正在分割文档...")
        
        # 使用递归字符分割器
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            separators=["\n## ", "\n# ", "\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"文档分割为 {len(chunks)} 个文本块")
        
        # 为每个块添加元数据
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["source"] = os.path.basename(chunk.metadata.get("source", ""))
        
        return chunks
    
    def prepare_data(self) -> List[Document]:
        """准备数据完整流程"""
        documents = self.load_recipes()
        chunks = self.split_documents(documents)
        return chunks
