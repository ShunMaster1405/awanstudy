"""
文档处理模块
"""

import os
import re
from typing import List, Dict
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import Config

class DocumentProcessor:
    """文档处理模块"""
    
    def __init__(self):
        self.config = Config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )
    
    def load_documents(self) -> List[Document]:
        """加载文档"""
        print("加载文档...")
        
        documents = []
        data_path = self.config.DATA_PATH
        
        if not os.path.exists(data_path):
            print(f"数据目录不存在: {data_path}")
            return documents
        
        for filename in os.listdir(data_path):
            if filename.endswith('.txt'):
                filepath = os.path.join(data_path, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 提取标题（第一行）
                    lines = content.strip().split('\n')
                    title = lines[0].replace('#', '').strip() if lines else filename
                    
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": filename,
                            "title": title,
                            "filepath": filepath
                        }
                    )
                    documents.append(doc)
                    print(f"加载文档: {filename}")
                    
                except Exception as e:
                    print(f"加载文档失败 {filename}: {e}")
        
        print(f"共加载 {len(documents)} 个文档")
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档"""
        print("分割文档...")
        
        split_docs = []
        for doc in documents:
            chunks = self.text_splitter.split_documents([doc])
            for chunk in chunks:
                # 保留原始元数据
                chunk.metadata.update(doc.metadata)
                # 添加块ID
                chunk.metadata["chunk_id"] = f"{doc.metadata['source']}_{len(split_docs)}"
            split_docs.extend(chunks)
        
        print(f"文档分割完成，共 {len(split_docs)} 个块")
        return split_docs
    
    def preprocess_text(self, text: str) -> str:
        """预处理文本"""
        # 移除多余空格和换行
        text = re.sub(r'\s+', ' ', text)
        # 移除特殊字符但保留中文和英文
        text = re.sub(r'[^\w\u4e00-\u9fff\s.,!?;:]', ' ', text)
        return text.strip()
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """提取关键词（简单实现）"""
        # 移除停用词
        stop_words = {'的', '了', '在', '是', '和', '与', '或', '等', '这个', '那个', '一个'}
        
        # 分割词语
        words = re.findall(r'[\u4e00-\u9fff]{2,}|[A-Za-z]{3,}', text)
        
        # 统计词频
        word_freq = {}
        for word in words:
            if word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # 按频率排序
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # 返回前N个关键词
        keywords = [word for word, freq in sorted_words[:top_n]]
        
        return keywords
    
    def load_and_process_documents(self) -> List[Document]:
        """加载并处理文档"""
        # 加载文档
        documents = self.load_documents()
        
        if not documents:
            return []
        
        # 分割文档
        split_docs = self.split_documents(documents)
        
        # 预处理每个文档块
        processed_docs = []
        for doc in split_docs:
            processed_text = self.preprocess_text(doc.page_content)
            keywords = self.extract_keywords(processed_text)
            
            # 创建新的文档对象
            new_doc = Document(
                page_content=processed_text,
                metadata={
                    **doc.metadata,
                    "keywords": keywords,
                    "original_length": len(doc.page_content),
                    "processed_length": len(processed_text)
                }
            )
            processed_docs.append(new_doc)
        
        return processed_docs
    
    def create_sample_documents(self) -> List[Document]:
        """创建示例文档"""
        print("创建示例文档...")
        
        sample_documents = [
            {
                "title": "人工智能概述",
                "content": """
                人工智能是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的智能机器。
                人工智能包括机器学习、深度学习、自然语言处理、计算机视觉等多个子领域。
                人工智能技术已经广泛应用于医疗、金融、教育、交通等各个行业。
                """
            },
            {
                "title": "机器学习基础",
                "content": """
                机器学习是人工智能的一个子领域，使计算机能够在没有明确编程的情况下学习和改进。
                机器学习主要分为监督学习、无监督学习和强化学习三种类型。
                监督学习使用带标签的数据进行训练，无监督学习发现数据中的模式，强化学习通过试错进行学习。
                """
            },
            {
                "title": "深度学习应用",
                "content": """
                深度学习是机器学习的一个子集，使用神经网络模拟人脑的工作方式。
                深度学习在计算机视觉、语音识别、自然语言处理等领域有广泛应用。
                常见的深度学习模型包括卷积神经网络、循环神经网络和Transformer。
                """
            },
            {
                "title": "自然语言处理技术",
                "content": """
                自然语言处理是人工智能的一个重要分支，专注于计算机与人类语言之间的交互。
                NLP技术包括文本分类、情感分析、机器翻译、问答系统等。
                大语言模型如GPT系列在NLP任务上表现出色。
                """
            },
            {
                "title": "计算机视觉进展",
                "content": """
                计算机视觉使计算机能够从图像和视频中获取信息并做出决策。
                计算机视觉应用包括图像分类、目标检测、人脸识别、自动驾驶等。
                深度学习极大地推动了计算机视觉的发展。
                """
            }
        ]
        
        # 确保数据目录存在
        os.makedirs(self.config.DATA_PATH, exist_ok=True)
        
        documents = []
        for i, doc_info in enumerate(sample_documents, 1):
            filename = f"sample_doc_{i}.txt"
            filepath = os.path.join(self.config.DATA_PATH, filename)
            
            # 写入文件
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# {doc_info['title']}\n\n")
                f.write(doc_info['content'])
            
            # 创建文档对象
            doc = Document(
                page_content=doc_info['content'],
                metadata={
                    "source": filename,
                    "title": doc_info['title'],
                    "filepath": filepath
                }
            )
            documents.append(doc)
            print(f"创建示例文档: {filename}")
        
        print(f"创建了 {len(documents)} 个示例文档")
        
        # 处理文档
        return self.load_and_process_documents()
