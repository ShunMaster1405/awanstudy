"""
知识抽取模块
"""

import os
import re
from typing import List, Dict, Tuple
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import Config

class KnowledgeExtraction:
    """知识抽取模块"""
    
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
                            "title": title
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
            split_docs.extend(chunks)
        
        print(f"文档分割完成，共 {len(split_docs)} 个块")
        return split_docs
    
    def extract_entities(self, text: str) -> List[str]:
        """提取实体（通用版）- 结合规则和统计方法"""
        entities = []
        
        # 1. 基于规则的提取
        # 提取中文名词短语（2-6个字符）
        chinese_noun_pattern = r'[\u4e00-\u9fff]{2,6}'
        noun_matches = re.findall(chinese_noun_pattern, text)
        entities.extend(noun_matches)
        
        # 2. 提取被特殊符号标记的术语
        special_patterns = [
            r'【([^】]+)】',      # 中文方括号
            r'「([^」]+)」',      # 中文引号
            r'《([^》]+)》',      # 中文书名号
            r'\(([^)]+)\)',      # 英文括号
            r'\[([^\]]+)\]',     # 英文方括号
        ]
        
        for pattern in special_patterns:
            matches = re.findall(pattern, text)
            entities.extend(matches)
        
        # 3. 提取定义性术语（X是Y，X包括Y等模式）
        definition_patterns = [
            r'([\u4e00-\u9fff]{2,8})是[\u4e00-\u9fff]+',    # X是Y
            r'([\u4e00-\u9fff]{2,8})包括[\u4e00-\u9fff]+',  # X包括Y
            r'([\u4e00-\u9fff]{2,8})称为[\u4e00-\u9fff]+',  # X称为Y
            r'([\u4e00-\u9fff]{2,8})指[\u4e00-\u9fff]+',    # X指Y
            r'([\u4e00-\u9fff]{2,8})即[\u4e00-\u9fff]+',    # X即Y
        ]
        
        for pattern in definition_patterns:
            matches = re.findall(pattern, text)
            entities.extend(matches)
        
        # 4. 提取英文术语（大写字母、缩写等）
        english_patterns = [
            r'\b[A-Z]{2,}\b',           # 大写缩写（AI, ML, DL等）
            r'\b[A-Z][a-z]+\b',         # 首字母大写的单词
            r'\b[a-z]+[A-Z][a-z]+\b',   # 驼峰命名
        ]
        
        for pattern in english_patterns:
            matches = re.findall(pattern, text)
            entities.extend(matches)
        
        # 5. 去重
        entities = list(set(entities))
        
        # 6. 过滤和评分
        filtered_entities = []
        for entity in entities:
            score = self._score_entity(entity, text)
            if score >= 2:  # 分数阈值
                filtered_entities.append((entity, score))
        
        # 7. 按分数排序并返回
        filtered_entities.sort(key=lambda x: x[1], reverse=True)
        result = [entity for entity, score in filtered_entities[:20]]  # 返回前20个
        
        return result
    
    def _score_entity(self, entity: str, text: str) -> int:
        """给实体评分"""
        score = 0
        
        # 基本规则
        if len(entity) >= 2:
            score += 1
        
        # 排除常见停用词
        stop_words = {'的', '了', '在', '是', '和', '与', '或', '等', 
                     '这个', '那个', '一个', '一种', '包括', '可以'}
        if entity in stop_words:
            return 0
        
        # 检查是否在特殊上下文中出现
        # 1. 是否被特殊符号包围
        if f'【{entity}】' in text or f'「{entity}」' in text or f'《{entity}》' in text:
            score += 3
        
        # 2. 是否在定义性上下文中
        definition_patterns = [
            f'{entity}是',
            f'{entity}包括',
            f'{entity}称为',
            f'{entity}指',
            f'{entity}即',
        ]
        
        for pattern in definition_patterns:
            if pattern in text:
                score += 2
        
        # 3. 是否多次出现（重要性指标）
        count = text.count(entity)
        if count >= 2:
            score += min(count, 5)  # 最多加5分
        
        # 4. 是否是英文术语（通常更重要）
        if re.match(r'^[A-Za-z]+$', entity):
            score += 2
            if entity.isupper() and len(entity) >= 2:  # 缩写
                score += 1
        
        return score
    
    def extract_relations(self, text: str, entities: List[str]) -> List[Tuple[str, str, str]]:
        """提取关系（改进版）"""
        relations = []
        
        # 改进的关系模式，针对AI领域
        patterns = [
            (r'([\u4e00-\u9fff]+)是([\u4e00-\u9fff]+)', '是', '定义关系'),
            (r'([\u4e00-\u9fff]+)的一个([\u4e00-\u9fff]+)', '包含', '层级关系'),
            (r'([\u4e00-\u9fff]+)使用([\u4e00-\u9fff]+)', '使用', '技术关系'),
            (r'([\u4e00-\u9fff]+)基于([\u4e00-\u9fff]+)', '基于', '依赖关系'),
            (r'([\u4e00-\u9fff]+)模拟([\u4e00-\u9fff]+)', '模拟', '模仿关系'),
            (r'([\u4e00-\u9fff]+)执行([\u4e00-\u9fff]+)', '执行', '功能关系'),
            (r'([\u4e00-\u9fff]+)创建([\u4e00-\u9fff]+)', '创建', '创造关系'),
        ]
        
        for pattern, relation_type, description in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) == 2:
                    entity1, entity2 = match
                    # 检查实体是否在实体列表中或是否是有意义的术语
                    if (entity1 in entities or len(entity1) >= 2) and (entity2 in entities or len(entity2) >= 2):
                        # 过滤掉无意义的词
                        if entity1 not in ['一个', '这个', '那个', '旨在', '能够'] and \
                           entity2 not in ['一个', '这个', '那个', '任务', '情况']:
                            relations.append((entity1, relation_type, entity2))
        
        # 如果没有找到关系，尝试基于实体之间的语义创建一些关系
        if not relations and len(entities) >= 2:
            # 创建一些基于常见AI知识的关系
            ai_knowledge = {
                '人工智能': ['计算机科学', '分支'],
                '机器学习': ['人工智能', '子领域'],
                '深度学习': ['机器学习', '子集'],
                '神经网络': ['深度学习', '使用'],
                '计算机': ['机器学习', '使用'],
            }
            
            for entity in entities:
                if entity in ai_knowledge:
                    for related_entity, rel_type in [(ai_knowledge[entity][0], '属于'), 
                                                     (ai_knowledge[entity][1], '是')]:
                        if related_entity in entities:
                            relations.append((entity, rel_type, related_entity))
        
        return relations
    
    def prepare_knowledge_data(self) -> Dict:
        """准备知识数据"""
        print("准备知识数据...")
        
        # 加载文档
        documents = self.load_documents()
        if not documents:
            print("没有找到文档，使用示例数据")
            documents = self._create_sample_documents()
        
        # 分割文档
        split_docs = self.split_documents(documents)
        
        # 提取知识和实体
        knowledge_data = {
            "documents": split_docs,
            "entities": [],
            "relations": []
        }
        
        # 从每个文档块中提取实体和关系
        for doc in split_docs:
            text = doc.page_content
            entities = self.extract_entities(text)
            relations = self.extract_relations(text, entities)
            
            knowledge_data["entities"].extend(entities)
            knowledge_data["relations"].extend(relations)
        
        # 去重
        knowledge_data["entities"] = list(set(knowledge_data["entities"]))
        
        print(f"知识数据准备完成:")
        print(f"  - 文档块: {len(knowledge_data['documents'])}")
        print(f"  - 实体: {len(knowledge_data['entities'])}")
        print(f"  - 关系: {len(knowledge_data['relations'])}")
        
        return knowledge_data
    
    def _create_sample_documents(self) -> List[Document]:
        """创建示例文档"""
        sample_texts = [
            {
                "title": "人工智能概述",
                "content": "人工智能是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的智能机器。"
            },
            {
                "title": "机器学习",
                "content": "机器学习是人工智能的一个子领域，使计算机能够在没有明确编程的情况下学习和改进。"
            },
            {
                "title": "深度学习",
                "content": "深度学习是机器学习的一个子集，使用神经网络模拟人脑的工作方式。"
            }
        ]
        
        documents = []
        for i, text in enumerate(sample_texts, 1):
            doc = Document(
                page_content=f"# {text['title']}\n\n{text['content']}",
                metadata={
                    "source": f"document_{i}.txt",
                    "title": text['title']
                }
            )
            documents.append(doc)
        
        return documents
