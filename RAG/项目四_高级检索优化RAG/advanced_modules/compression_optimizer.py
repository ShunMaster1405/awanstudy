"""
压缩优化器模块
"""

import re
from typing import List, Dict, Any
from langchain.schema import Document
from config import Config

class CompressionOptimizer:
    """压缩优化器 - 支持多种压缩技术"""
    
    def __init__(self):
        self.config = Config
    
    def filter_by_relevance(self, documents: List[Document], threshold: float = 0.3) -> List[Document]:
        """基于相关性过滤文档"""
        print(f"基于相关性过滤文档 (阈值: {threshold})...")
        
        filtered_docs = []
        
        for doc in documents:
            # 从元数据中获取相关性分数
            score = doc.metadata.get("relevance_score", 0.0)
            
            if score >= threshold:
                filtered_docs.append(doc)
        
        print(f"过滤结果: {len(documents)} -> {len(filtered_docs)} 个文档")
        return filtered_docs
    
    def extract_key_sentences(self, document: Document, query: str = None, max_sentences: int = 3) -> Document:
        """提取关键句子"""
        content = document.page_content
        
        # 分割句子
        sentences = self._split_sentences(content)
        
        if not sentences:
            return document
        
        # 如果没有查询，选择前几个句子
        if not query:
            selected_sentences = sentences[:max_sentences]
        else:
            # 基于查询相关性选择句子
            selected_sentences = self._select_sentences_by_query(sentences, query, max_sentences)
        
        # 构建压缩后的内容
        compressed_content = " ".join(selected_sentences)
        
        # 创建新的文档对象
        compressed_doc = Document(
            page_content=compressed_content,
            metadata={
                **document.metadata,
                "compression_method": "key_sentence_extraction",
                "original_length": len(content),
                "compressed_length": len(compressed_content),
                "compression_ratio": len(compressed_content) / max(len(content), 1)
            }
        )
        
        return compressed_doc
    
    def _split_sentences(self, text: str) -> List[str]:
        """分割句子"""
        # 使用标点符号分割句子
        sentences = re.split(r'[。！？.!?]', text)
        
        # 清理句子
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _select_sentences_by_query(self, sentences: List[str], query: str, max_sentences: int) -> List[str]:
        """基于查询选择句子"""
        # 简单关键词匹配
        query_words = set(query.lower().split())
        
        # 计算每个句子的相关性分数
        sentence_scores = []
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            
            # 计算Jaccard相似度
            intersection = len(query_words.intersection(sentence_words))
            union = len(query_words.union(sentence_words))
            similarity = intersection / max(union, 1)
            
            sentence_scores.append((sentence, similarity))
        
        # 按相似度排序
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 选择前几个句子
        selected_sentences = [sentence for sentence, score in sentence_scores[:max_sentences]]
        
        # 如果选择的句子太少，补充一些原始句子
        if len(selected_sentences) < max_sentences:
            remaining = max_sentences - len(selected_sentences)
            for sentence in sentences:
                if sentence not in selected_sentences:
                    selected_sentences.append(sentence)
                    remaining -= 1
                    if remaining <= 0:
                        break
        
        return selected_sentences
    
    def generate_summary(self, document: Document, max_length: int = 200) -> Document:
        """生成摘要"""
        content = document.page_content
        
        # 简单摘要生成（取前几个句子）
        sentences = self._split_sentences(content)
        
        if not sentences:
            return document
        
        # 选择句子直到达到最大长度
        summary_sentences = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            if current_length + sentence_length <= max_length:
                summary_sentences.append(sentence)
                current_length += sentence_length
            else:
                break
        
        # 构建摘要
        summary = " ".join(summary_sentences)
        
        # 如果摘要太短，添加更多内容
        if len(summary) < max_length // 2 and len(sentences) > len(summary_sentences):
            # 添加下一个句子的一部分
            next_sentence = sentences[len(summary_sentences)]
            remaining_length = max_length - len(summary)
            if remaining_length > 20:  # 至少保留20个字符
                summary += " " + next_sentence[:remaining_length] + "..."
        
        # 创建新的文档对象
        summarized_doc = Document(
            page_content=summary,
            metadata={
                **document.metadata,
                "compression_method": "summary_generation",
                "original_length": len(content),
                "compressed_length": len(summary),
                "compression_ratio": len(summary) / max(len(content), 1),
                "max_length": max_length
            }
        )
        
        return summarized_doc
    
    def semantic_compression(self, document: Document, query: str = None) -> Document:
        """语义压缩"""
        content = document.page_content
        
        # 提取关键词
        keywords = self._extract_keywords(content)
        
        # 基于关键词重建内容
        if query:
            # 如果有关键词，优先包含与查询相关的关键词
            query_words = set(query.lower().split())
            relevant_keywords = [kw for kw in keywords if kw.lower() in query_words]
            
            # 添加其他关键词
            other_keywords = [kw for kw in keywords if kw.lower() not in query_words][:5]
            
            all_keywords = relevant_keywords + other_keywords
        else:
            all_keywords = keywords[:10]
        
        # 构建语义摘要
        if all_keywords:
            semantic_summary = f"关键词: {', '.join(all_keywords)}. "
            
            # 添加一些上下文
            sentences = self._split_sentences(content)
            if sentences:
                # 选择包含关键词的句子
                keyword_sentences = []
                for sentence in sentences[:3]:  # 只看前3个句子
                    for keyword in all_keywords:
                        if keyword in sentence:
                            keyword_sentences.append(sentence)
                            break
                
                if keyword_sentences:
                    semantic_summary += " ".join(keyword_sentences[:2])
                else:
                    semantic_summary += sentences[0]
        else:
            # 如果没有关键词，使用第一个句子
            sentences = self._split_sentences(content)
            semantic_summary = sentences[0] if sentences else content[:100]
        
        # 创建新的文档对象
        compressed_doc = Document(
            page_content=semantic_summary,
            metadata={
                **document.metadata,
                "compression_method": "semantic_compression",
                "original_length": len(content),
                "compressed_length": len(semantic_summary),
                "compression_ratio": len(semantic_summary) / max(len(content), 1),
                "keywords": keywords[:10]
            }
        )
        
        return compressed_doc
    
    def _extract_keywords(self, text: str, top_n: int = 15) -> List[str]:
        """提取关键词"""
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
    
    def compression_pipeline(self, documents: List[Document], query: str = None) -> List[Document]:
        """压缩管道"""
        print("=" * 50)
        print("压缩管道")
        print("=" * 50)
        
        if not documents:
            print("没有文档可压缩")
            return []
        
        print(f"输入文档数: {len(documents)}")
        print(f"查询: {query if query else '无'}")
        
        compressed_docs = []
        
        for i, doc in enumerate(documents):
            print(f"\n处理文档 {i+1}/{len(documents)}: {doc.metadata.get('title', '未知标题')}")
            
            # 应用多种压缩技术
            print("  - 应用压缩技术...")
            
            # 1. 语义压缩
            semantic_doc = self.semantic_compression(doc, query)
            
            # 2. 关键句子提取
            key_sentence_doc = self.extract_key_sentences(semantic_doc, query)
            
            # 3. 摘要生成
            final_doc = self.generate_summary(key_sentence_doc)
            
            # 记录压缩统计
            original_length = doc.metadata.get("original_length", len(doc.page_content))
            compressed_length = len(final_doc.page_content)
            compression_ratio = compressed_length / max(original_length, 1)
            
            print(f"    原始长度: {original_length}")
            print(f"    压缩后长度: {compressed_length}")
            print(f"    压缩比: {compression_ratio:.2%}")
            
            compressed_docs.append(final_doc)
        
        # 计算总体统计
        total_original = sum(len(doc.page_content) for doc in documents)
        total_compressed = sum(len(doc.page_content) for doc in compressed_docs)
        overall_ratio = total_compressed / max(total_original, 1)
        
        print(f"\n压缩完成:")
        print(f"  总原始长度: {total_original}")
        print(f"  总压缩后长度: {total_compressed}")
        print(f"  总体压缩比: {overall_ratio:.2%}")
        print(f"  文档数量: {len(compressed_docs)}")
        
        return compressed_docs
