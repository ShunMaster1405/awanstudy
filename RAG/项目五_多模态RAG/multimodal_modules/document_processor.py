"""
多模态文档处理器
"""

import os
import re
from typing import List, Dict, Any
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import Config

class MultimodalDocumentProcessor:
    """多模态文档处理器 - 支持文本和图像处理"""
    
    def __init__(self):
        self.config = Config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )
    
    def load_documents(self) -> List[Document]:
        """加载多模态文档"""
        print("加载多模态文档...")
        
        documents = []
        
        # 加载文本文档
        text_docs = self._load_text_documents()
        documents.extend(text_docs)
        
        # 加载图像文档（元数据）
        image_docs = self._load_image_documents()
        documents.extend(image_docs)
        
        print(f"共加载 {len(documents)} 个多模态文档")
        return documents
    
    def _load_text_documents(self) -> List[Document]:
        """加载文本文档"""
        documents = []
        text_path = self.config.TEXT_DATA_PATH
        
        if not os.path.exists(text_path):
            print(f"文本数据路径不存在: {text_path}")
            return documents
        
        # 读取所有txt文件
        for filename in os.listdir(text_path):
            if filename.endswith('.txt'):
                filepath = os.path.join(text_path, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 提取标题
                    title = self._extract_title(content, filename)
                    
                    # 创建文档对象
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": filename,
                            "title": title,
                            "filepath": filepath,
                            "modality": "text",
                            "content_type": "text"
                        }
                    )
                    documents.append(doc)
                    print(f"加载文本文档: {filename}")
                    
                except Exception as e:
                    print(f"加载文本文档失败 {filename}: {e}")
        
        return documents
    
    def _load_image_documents(self) -> List[Document]:
        """加载图像文档（创建图像描述）"""
        documents = []
        image_path = self.config.IMAGE_DATA_PATH
        
        if not os.path.exists(image_path):
            print(f"图像数据路径不存在: {image_path}")
            return documents
        
        # 创建示例图像描述（实际应用中需要图像分析模型）
        sample_image_descriptions = [
            {
                "filename": "ai_vision.jpg",
                "description": "人工智能视觉识别系统示意图，展示计算机视觉技术在图像识别、物体检测和场景理解方面的应用。",
                "tags": ["人工智能", "计算机视觉", "图像识别", "深度学习"]
            },
            {
                "filename": "multimodal_learning.png",
                "description": "多模态学习架构图，展示文本、图像、音频等多种模态数据的融合处理流程。",
                "tags": ["多模态学习", "数据融合", "跨模态", "深度学习"]
            },
            {
                "filename": "clip_model.jpg",
                "description": "CLIP模型架构示意图，展示对比语言-图像预训练模型的工作原理和结构。",
                "tags": ["CLIP", "对比学习", "视觉语言模型", "预训练"]
            }
        ]
        
        # 检查实际图像文件
        image_files = []
        for filename in os.listdir(image_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                image_files.append(filename)
        
        # 如果有实际图像文件，使用实际文件名
        if image_files:
            for i, filename in enumerate(image_files[:3]):  # 最多处理3个图像
                filepath = os.path.join(image_path, filename)
                
                # 创建图像描述
                description = f"图像文件: {filename}。这是一个多模态RAG系统中的图像数据，包含视觉信息。"
                
                doc = Document(
                    page_content=description,
                    metadata={
                        "source": filename,
                        "title": f"图像_{i+1}",
                        "filepath": filepath,
                        "modality": "image",
                        "content_type": "image",
                        "description": description
                    }
                )
                documents.append(doc)
                print(f"加载图像文档: {filename}")
        else:
            # 使用示例描述
            for i, sample in enumerate(sample_image_descriptions):
                doc = Document(
                    page_content=sample["description"],
                    metadata={
                        "source": sample["filename"],
                        "title": f"图像_{i+1}",
                        "filepath": os.path.join(image_path, sample["filename"]),
                        "modality": "image",
                        "content_type": "image",
                        "description": sample["description"],
                        "tags": sample["tags"]
                    }
                )
                documents.append(doc)
                print(f"创建图像文档示例: {sample['filename']}")
        
        return documents
    
    def _extract_title(self, content: str, filename: str) -> str:
        """从内容中提取标题"""
        lines = content.strip().split('\n')
        if lines:
            first_line = lines[0].strip()
            # 如果第一行看起来像标题
            if 2 <= len(first_line) <= 50 and not first_line.endswith(('.', '。', '!', '！', '?', '？')):
                return first_line
        
        # 使用文件名作为标题
        title = os.path.splitext(filename)[0]
        title = title.replace('_', ' ').replace('-', ' ')
        return title
    
    def preprocess_documents(self, documents: List[Document]) -> List[Document]:
        """预处理多模态文档"""
        print("预处理多模态文档...")
        
        processed_docs = []
        
        for i, doc in enumerate(documents):
            # 根据模态类型进行不同处理
            modality = doc.metadata.get("modality", "text")
            
            if modality == "text":
                processed_doc = self._preprocess_text_document(doc, i)
            else:  # image
                processed_doc = self._preprocess_image_document(doc, i)
            
            processed_docs.append(processed_doc)
        
        print(f"预处理完成: {len(processed_docs)} 个文档")
        return processed_docs
    
    def _preprocess_text_document(self, doc: Document, index: int) -> Document:
        """预处理文本文档"""
        # 清理文本
        cleaned_content = self._clean_text(doc.page_content)
        
        # 提取关键词
        keywords = self._extract_keywords(cleaned_content)
        
        # 计算文本特征
        features = self._calculate_text_features(cleaned_content)
        
        # 创建新的文档对象
        processed_doc = Document(
            page_content=cleaned_content,
            metadata={
                **doc.metadata,
                "chunk_id": f"text_chunk_{index:03d}",
                "keywords": keywords,
                "features": features,
                "length": len(cleaned_content),
                "word_count": len(cleaned_content.split()),
                "modality": "text"
            }
        )
        
        return processed_doc
    
    def _preprocess_image_document(self, doc: Document, index: int) -> Document:
        """预处理图像文档"""
        content = doc.page_content
        
        # 提取图像特征（简化版）
        features = {
            "modality": "image",
            "has_description": bool(content.strip()),
            "description_length": len(content),
            "estimated_objects": self._estimate_image_objects(content)
        }
        
        # 创建新的文档对象
        processed_doc = Document(
            page_content=content,
            metadata={
                **doc.metadata,
                "chunk_id": f"image_chunk_{index:03d}",
                "features": features,
                "modality": "image"
            }
        )
        
        return processed_doc
    
    def _clean_text(self, text: str) -> str:
        """清理文本"""
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        text = re.sub(r'[^\w\u4e00-\u9fff\s.,!?;:，。！？；：]', '', text)
        return text
    
    def _extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """提取关键词"""
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
            if len(word) > 1:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # 按频率排序
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # 返回top_n关键词
        keywords = [word for word, freq in sorted_words[:top_n]]
        return keywords
    
    def _calculate_text_features(self, text: str) -> Dict[str, Any]:
        """计算文本特征"""
        char_count = len(text)
        word_count = len(text.split())
        sentence_count = len(re.split(r'[。！？.!?]', text))
        
        words = text.split()
        unique_words = set(words)
        lexical_density = len(unique_words) / max(len(words), 1)
        
        avg_word_length = sum(len(word) for word in words) / max(len(words), 1)
        
        return {
            "char_count": char_count,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "lexical_density": lexical_density,
            "avg_word_length": avg_word_length,
            "modality": "text"
        }
    
    def _estimate_image_objects(self, description: str) -> List[str]:
        """从图像描述中估计物体"""
        # 简单的关键词提取
        objects = []
        common_objects = ["人", "车", "建筑", "动物", "植物", "电脑", "手机", "书", "桌子", "椅子"]
        
        for obj in common_objects:
            if obj in description:
                objects.append(obj)
        
        return objects if objects else ["通用物体"]
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档（只分割文本文档）"""
        print("分割文档...")
        
        if not documents:
            print("没有文档可分割")
            return []
        
        split_docs = []
        
        for doc in documents:
            modality = doc.metadata.get("modality", "text")
            
            if modality == "text":
                # 分割文本文档
                text_splits = self.text_splitter.split_documents([doc])
                split_docs.extend(text_splits)
            else:
                # 图像文档不分割
                split_docs.append(doc)
        
        # 为每个分块添加元数据
        for i, doc in enumerate(split_docs):
            doc.metadata["chunk_id"] = f"chunk_{i:03d}"
            doc.metadata["chunk_index"] = i
            doc.metadata["total_chunks"] = len(split_docs)
        
        print(f"文档分割完成，共 {len(split_docs)} 个块")
        return split_docs
    
    def process_pipeline(self) -> List[Document]:
        """完整的多模态文档处理管道"""
        print("=" * 50)
        print("多模态文档处理管道")
        print("=" * 50)
        
        # 1. 加载文档
        raw_documents = self.load_documents()
        
        if not raw_documents:
            print("错误：没有找到多模态文档文件")
            print(f"请确保在 {self.config.TEXT_DATA_PATH} 和 {self.config.IMAGE_DATA_PATH} 目录下放置文档文件")
            return []
        
        # 2. 预处理文档
        preprocessed_docs = self.preprocess_documents(raw_documents)
        
        # 3. 分割文档
        split_docs = self.split_documents(preprocessed_docs)
        
        print(f"多模态文档处理完成: {len(split_docs)} 个文档块")
        return split_docs
