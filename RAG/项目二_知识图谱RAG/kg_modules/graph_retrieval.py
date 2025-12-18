"""
图检索模块（简化版）
"""

from typing import List, Dict, Any
from langchain.schema import Document

class GraphRetrieval:
    """图检索模块"""
    
    def __init__(self, graph_data: Dict):
        self.graph_data = graph_data
    
    def retrieve_from_graph(self, query: str, k: int = 3) -> List[Document]:
        """从图中检索相关信息（简化版）"""
        print(f"图检索查询: {query}")
        
        results = []
        
        # 从图数据中检索
        entities = self.graph_data.get("entities", [])
        relations = self.graph_data.get("relations", [])
        
        # 查找相关实体
        query_terms = query.lower().split()
        relevant_entities = []
        
        for entity in entities:
            entity_lower = entity.lower()
            for term in query_terms:
                if term in entity_lower:
                    relevant_entities.append(entity)
                    break
        
        # 查找相关关系
        relevant_relations = []
        for relation in relations:
            if len(relation) == 3:
                entity1, rel_type, entity2 = relation
                # 检查关系是否与查询相关
                for term in query_terms:
                    if (term in entity1.lower() or 
                        term in entity2.lower() or 
                        term in rel_type.lower()):
                        relevant_relations.append(relation)
                        break
        
        # 创建文档结果
        if relevant_entities:
            entities_text = "相关实体: " + ", ".join(relevant_entities[:5])
            results.append(Document(
                page_content=entities_text,
                metadata={"source": "graph_entities", "type": "entities"}
            ))
        
        if relevant_relations:
            relations_text = "相关关系:\n"
            for i, (e1, rel, e2) in enumerate(relevant_relations[:3]):
                relations_text += f"{i+1}. {e1} - {rel} - {e2}\n"
            
            results.append(Document(
                page_content=relations_text.strip(),
                metadata={"source": "graph_relations", "type": "relations"}
            ))
        
        # 如果没有找到相关结果，返回一些通用信息
        if not results:
            graph_info = f"知识图谱信息:\n"
            graph_info += f"- 实体数量: {len(entities)}\n"
            graph_info += f"- 关系数量: {len(relations)}\n"
            if entities:
                graph_info += f"- 示例实体: {', '.join(entities[:3])}\n"
            if relations:
                graph_info += f"- 示例关系: {relations[0][0]} - {relations[0][1]} - {relations[0][2]}"
            
            results.append(Document(
                page_content=graph_info,
                metadata={"source": "graph_summary", "type": "summary"}
            ))
        
        print(f"图检索找到 {len(results)} 个相关结果")
        return results[:k]
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """获取图统计信息"""
        return {
            "num_entities": len(self.graph_data.get("entities", [])),
            "num_relations": len(self.graph_data.get("relations", [])),
            "entities_sample": self.graph_data.get("entities", [])[:5],
            "relations_sample": self.graph_data.get("relations", [])[:3]
        }
    
    def find_entity_paths(self, entity1: str, entity2: str, max_depth: int = 2) -> List[List[str]]:
        """查找实体间的路径（简化版）"""
        # 注意：这是一个简化版，实际应该使用图算法
        # 这里我们只返回直接关系
        
        paths = []
        relations = self.graph_data.get("relations", [])
        
        # 查找直接关系
        for rel in relations:
            if len(rel) == 3:
                e1, rel_type, e2 = rel
                if (e1 == entity1 and e2 == entity2) or (e1 == entity2 and e2 == entity1):
                    paths.append([entity1, rel_type, entity2])
        
        # 如果没有直接关系，尝试查找间接关系
        if not paths and max_depth > 1:
            # 查找entity1的所有关系
            entity1_relations = []
            for rel in relations:
                if len(rel) == 3:
                    e1, rel_type, e2 = rel
                    if e1 == entity1:
                        entity1_relations.append((e2, rel_type))
                    elif e2 == entity1:
                        entity1_relations.append((e1, rel_type))
            
            # 查找entity2的所有关系
            entity2_relations = []
            for rel in relations:
                if len(rel) == 3:
                    e1, rel_type, e2 = rel
                    if e1 == entity2:
                        entity2_relations.append((e2, rel_type))
                    elif e2 == entity2:
                        entity2_relations.append((e1, rel_type))
            
            # 查找共同邻居
            for neighbor1, rel1 in entity1_relations:
                for neighbor2, rel2 in entity2_relations:
                    if neighbor1 == neighbor2:
                        paths.append([entity1, rel1, neighbor1, rel2, entity2])
        
        return paths[:5]  # 限制返回数量
