"""
文档处理模块 - 高级检索优化版本
"""

import os
import re
from typing import List, Dict, Any
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import Config

class DocumentProcessor:
    """文档处理器 - 支持高级预处理和优化"""
    
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
            print(f"数据路径不存在: {data_path}")
            return documents
        
        # 读取所有txt文件
        for filename in os.listdir(data_path):
            if filename.endswith('.txt'):
                filepath = os.path.join(data_path, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 提取标题（从第一行或文件名）
                    title = self._extract_title(content, filename)
                    
                    # 创建文档对象
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
    
    def _extract_title(self, content: str, filename: str) -> str:
        """从内容中提取标题"""
        # 尝试从第一行提取标题
        lines = content.strip().split('\n')
        if lines:
            first_line = lines[0].strip()
            # 如果第一行看起来像标题（长度适中，没有标点结尾）
            if 2 <= len(first_line) <= 50 and not first_line.endswith(('.', '。', '!', '！', '?', '？')):
                return first_line
        
        # 使用文件名作为标题
        title = os.path.splitext(filename)[0]
        # 美化文件名（替换下划线等）
        title = title.replace('_', ' ').replace('-', ' ')
        return title
    
    def preprocess_documents(self, documents: List[Document]) -> List[Document]:
        """预处理文档"""
        print("预处理文档...")
        
        processed_docs = []
        
        for i, doc in enumerate(documents):
            # 清理文本
            cleaned_content = self._clean_text(doc.page_content)
            
            # 提取关键词
            keywords = self._extract_keywords(cleaned_content)
            
            # 计算文本特征
            features = self._calculate_features(cleaned_content)
            
            # 创建新的文档对象
            processed_doc = Document(
                page_content=cleaned_content,
                metadata={
                    **doc.metadata,
                    "chunk_id": f"chunk_{i:03d}",
                    "keywords": keywords,
                    "features": features,
                    "length": len(cleaned_content),
                    "word_count": len(cleaned_content.split())
                }
            )
            processed_docs.append(processed_doc)
        
        print(f"预处理完成: {len(processed_docs)} 个文档")
        return processed_docs
    
    def _clean_text(self, text: str) -> str:
        """清理文本"""
        # 移除多余的空格和换行
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # 移除特殊字符（保留中文、英文、数字和基本标点）
        text = re.sub(r'[^\w\u4e00-\u9fff\s.,!?;:，。！？；：]', '', text)
        
        return text
    
    def _extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """提取关键词（简化版）"""
        # 中文按字符分割，英文按单词分割
        words = []
        current_word = ""
        
        for char in text:
            if '\u4e00' <= char <= '\u9fff':  # 中文字符
                if current_word:
                    words.append(current_word.lower())
                    current_word = ""
                words.append(char)
            elif char.isalpha():  # 英文字母
                current_word += char
            else:  # 其他字符
                if current_word:
                    words.append(current_word.lower())
                    current_word = ""
        
        if current_word:
            words.append(current_word.lower())
        
        # 统计词频
        word_freq = {}
        for word in words:
            if len(word) > 1:  # 只考虑长度大于1的词
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # 按频率排序
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # 返回top_n关键词
        keywords = [word for word, freq in sorted_words[:top_n]]
        return keywords
    
    def _calculate_features(self, text: str) -> Dict[str, Any]:
        """计算文本特征"""
        # 计算基本特征
        char_count = len(text)
        word_count = len(text.split())
        sentence_count = len(re.split(r'[。！？.!?]', text))
        
        # 计算词汇密度（唯一词比例）
        words = text.split()
        unique_words = set(words)
        lexical_density = len(unique_words) / max(len(words), 1)
        
        # 计算平均词长
        avg_word_length = sum(len(word) for word in words) / max(len(words), 1)
        
        return {
            "char_count": char_count,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "lexical_density": lexical_density,
            "avg_word_length": avg_word_length
        }
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档"""
        print("分割文档...")
        
        if not documents:
            print("没有文档可分割")
            return []
        
        # 使用LangChain的分割器
        split_docs = self.text_splitter.split_documents(documents)
        
        # 为每个分块添加元数据
        for i, doc in enumerate(split_docs):
            doc.metadata["chunk_id"] = f"chunk_{i:03d}"
            doc.metadata["chunk_index"] = i
            doc.metadata["total_chunks"] = len(split_docs)
        
        print(f"文档分割完成，共 {len(split_docs)} 个块")
        return split_docs
    
    def process_pipeline(self) -> List[Document]:
        """完整的文档处理管道"""
        print("=" * 50)
        print("文档处理管道")
        print("=" * 50)
        
        # 1. 加载文档
        raw_documents = self.load_documents()
        
        if not raw_documents:
            print("错误：没有找到文档文件")
            print(f"请确保在 {self.config.DATA_PATH} 目录下放置文档文件")
            return []
        
        # 2. 预处理文档
        preprocessed_docs = self.preprocess_documents(raw_documents)
        
        # 3. 分割文档
        split_docs = self.split_documents(preprocessed_docs)
        
        print(f"文档处理完成: {len(split_docs)} 个文档块")
        return split_docs
