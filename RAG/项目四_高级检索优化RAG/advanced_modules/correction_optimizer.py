"""
校正优化器模块 - Corrective-RAG (C-RAG)
"""

import re
from typing import List, Dict, Any, Tuple
from langchain.schema import Document
from config import Config

class CorrectionOptimizer:
    """校正优化器 - 实现Corrective-RAG (C-RAG)"""
    
    def __init__(self):
        self.config = Config
    
    def evaluate_retrieval_quality(self, query: str, documents: List[Document]) -> Dict[str, Any]:
        """评估检索质量"""
        print("评估检索质量...")
        
        if not documents:
            return {
                "overall_score": 0.0,
                "relevance_score": 0.0,
                "coverage_score": 0.0,
                "consistency_score": 0.0,
                "information_gaps": ["没有检索到任何文档"],
                "quality_level": "poor"
            }
        
        # 1. 相关性评估
        relevance_score = self._evaluate_relevance(query, documents)
        
        # 2. 覆盖度评估
        coverage_score = self._evaluate_coverage(query, documents)
        
        # 3. 一致性评估
        consistency_score = self._evaluate_consistency(documents)
        
        # 4. 识别信息缺口
        information_gaps = self._identify_information_gaps(query, documents)
        
        # 计算总体分数
        overall_score = 0.4 * relevance_score + 0.3 * coverage_score + 0.3 * consistency_score
        
        # 确定质量等级
        if overall_score >= 0.7:
            quality_level = "good"
        elif overall_score >= 0.4:
            quality_level = "fair"
        else:
            quality_level = "poor"
        
        evaluation_result = {
            "overall_score": overall_score,
            "relevance_score": relevance_score,
            "coverage_score": coverage_score,
            "consistency_score": consistency_score,
            "information_gaps": information_gaps,
            "quality_level": quality_level,
            "num_documents": len(documents)
        }
        
        print(f"检索质量评估结果:")
        print(f"  总体分数: {overall_score:.3f} ({quality_level})")
        print(f"  相关性分数: {relevance_score:.3f}")
        print(f"  覆盖度分数: {coverage_score:.3f}")
        print(f"  一致性分数: {consistency_score:.3f}")
        print(f"  信息缺口: {len(information_gaps)} 个")
        
        return evaluation_result
    
    def _evaluate_relevance(self, query: str, documents: List[Document]) -> float:
        """评估相关性"""
        query_words = set(query.lower().split())
        
        total_similarity = 0.0
        valid_docs = 0
        
        for doc in documents:
            content = doc.page_content.lower()
            doc_words = set(content.split())
            
            # 计算Jaccard相似度
            intersection = len(query_words.intersection(doc_words))
            union = len(query_words.union(doc_words))
            
            if union > 0:
                similarity = intersection / union
                total_similarity += similarity
                valid_docs += 1
        
        # 平均相似度
        if valid_docs > 0:
            avg_similarity = total_similarity / valid_docs
        else:
            avg_similarity = 0.0
        
        return avg_similarity
    
    def _evaluate_coverage(self, query: str, documents: List[Document]) -> float:
        """评估覆盖度"""
        query_words = set(query.lower().split())
        
        if not query_words:
            return 0.0
        
        # 收集所有文档中的单词
        all_doc_words = set()
        for doc in documents:
            content = doc.page_content.lower()
            doc_words = set(content.split())
            all_doc_words.update(doc_words)
        
        # 计算查询词在文档中的覆盖比例
        covered_words = query_words.intersection(all_doc_words)
        coverage = len(covered_words) / len(query_words)
        
        return coverage
    
    def _evaluate_consistency(self, documents: List[Document]) -> float:
        """评估一致性"""
        if len(documents) <= 1:
            return 1.0  # 只有一个文档，一致性为完美
        
        # 提取每个文档的关键词
        all_keywords = []
        for doc in documents:
            keywords = doc.metadata.get("keywords", [])
            if keywords:
                all_keywords.append(set(keywords))
        
        if len(all_keywords) <= 1:
            return 0.5  # 没有足够的关键词信息
        
        # 计算关键词重叠度
        total_overlap = 0.0
        pair_count = 0
        
        for i in range(len(all_keywords)):
            for j in range(i + 1, len(all_keywords)):
                set1 = all_keywords[i]
                set2 = all_keywords[j]
                
                if set1 and set2:
                    intersection = len(set1.intersection(set2))
                    union = len(set1.union(set2))
                    
                    if union > 0:
                        overlap = intersection / union
                        total_overlap += overlap
                        pair_count += 1
        
        if pair_count > 0:
            avg_overlap = total_overlap / pair_count
        else:
            avg_overlap = 0.0
        
        return avg_overlap
    
    def _identify_information_gaps(self, query: str, documents: List[Document]) -> List[str]:
        """识别信息缺口"""
        query_words = set(query.lower().split())
        
        if not query_words:
            return ["查询为空"]
        
        # 收集所有文档中的单词
        all_doc_words = set()
        for doc in documents:
            content = doc.page_content.lower()
            doc_words = set(content.split())
            all_doc_words.update(doc_words)
        
        # 找出未覆盖的查询词
        uncovered_words = query_words - all_doc_words
        
        # 将未覆盖的词转换为信息缺口描述
        information_gaps = []
        for word in uncovered_words:
            if len(word) > 2:  # 只考虑长度大于2的词
                information_gaps.append(f"缺少关于'{word}'的信息")
        
        # 如果文档太少，添加通用缺口
        if len(documents) < 3:
            information_gaps.append("检索到的文档数量不足")
        
        return information_gaps
    
    def supplement_external_knowledge(self, query: str, information_gaps: List[str]) -> List[Document]:
        """补充外部知识"""
        print("补充外部知识...")
        
        if not information_gaps:
            print("  没有信息缺口，无需补充")
            return []
        
        print(f"  信息缺口: {len(information_gaps)} 个")
        
        # 基于信息缺口生成补充文档
        supplementary_docs = []
        
        for i, gap in enumerate(information_gaps[:3]):  # 最多补充3个文档
            # 提取关键词
            keywords = re.findall(r"['\"]([^'\"]+)['\"]", gap)
            if not keywords:
                keywords = gap.split()[:3]
            
            # 生成补充内容
            if "重排序" in query or "rerank" in query.lower():
                content = self._generate_reranking_supplement(keywords)
                title = "重排序技术补充"
            elif "压缩" in query or "compress" in query.lower():
                content = self._generate_compression_supplement(keywords)
                title = "压缩技术补充"
            elif "校正" in query or "correct" in query.lower() or "C-RAG" in query:
                content = self._generate_correction_supplement(keywords)
                title = "校正技术补充"
            else:
                content = self._generate_general_supplement(keywords, query)
                title = "通用知识补充"
            
            # 创建文档对象
            doc = Document(
                page_content=content,
                metadata={
                    "source": f"supplementary_{i+1}.txt",
                    "title": title,
                    "supplementary": True,
                    "information_gap": gap,
                    "keywords": keywords
                }
            )
            
            supplementary_docs.append(doc)
            print(f"  生成补充文档: {title}")
        
        return supplementary_docs
    
    def _generate_reranking_supplement(self, keywords: List[str]) -> str:
        """生成重排序技术补充内容"""
        base_content = """重排序技术是检索优化的重要环节，通过重新排序初步检索结果来提升精度。

常用重排序方法包括：
1. Cross-Encoder: 对查询和文档进行联合编码，计算精确的相关性分数
2. ColBERT: 使用晚期交互策略，平衡精度和效率
3. RankLLM: 利用大语言模型进行零样本重排序
4. MonoT5: 基于T5的序列到序列重排序模型

重排序的关键优势：
- 提升检索精度：通过更精细的相关性计算
- 改善用户体验：返回更相关的结果
- 支持复杂查询：处理语义复杂的查询需求"""
        
        if keywords:
            keyword_str = "、".join(keywords[:3])
            return f"关于{keyword_str}的重排序技术：\n\n{base_content}"
        else:
            return base_content
    
    def _generate_compression_supplement(self, keywords: List[str]) -> str:
        """生成压缩技术补充内容"""
        base_content = """上下文压缩技术通过减少输入长度来提高生成效率和质量。

主要压缩方法：
1. 文档过滤：基于相关性分数筛选文档
2. 内容提取：识别和提取关键句子或段落
3. 摘要生成：为长文档生成简洁摘要
4. 语义压缩：基于语义分析重组内容

压缩技术的应用场景：
- 处理长文档：减少输入token数量
- 提高生成质量：聚焦关键信息
- 降低计算成本：减少模型处理负担
- 改善响应速度：加快生成过程"""
        
        if keywords:
            keyword_str = "、".join(keywords[:3])
            return f"关于{keyword_str}的压缩技术：\n\n{base_content}"
        else:
            return base_content
    
    def _generate_correction_supplement(self, keywords: List[str]) -> str:
        """生成校正技术补充内容"""
        base_content = """Corrective-RAG (C-RAG) 通过评估和校正检索结果来提升RAG系统性能。

C-RAG核心流程：
1. 检索质量评估：分析相关性、覆盖度、一致性
2. 外部知识补充：当检索质量不足时补充信息
3. 检索结果校正：优化排序、修正内容、补充信息
4. 增强生成：基于校正后的结果生成答案

C-RAG的优势：
- 自我校正：自动识别和修复检索问题
- 知识补充：动态补充缺失信息
- 质量保证：确保检索结果的质量
- 适应性：适应不同查询和文档类型"""
        
        if keywords:
            keyword_str = "、".join(keywords[:3])
            return f"关于{keyword_str}的校正技术：\n\n{base_content}"
        else:
            return base_content
    
    def _generate_general_supplement(self, keywords: List[str], query: str) -> str:
        """生成通用补充内容"""
        if keywords:
            keyword_str = "、".join(keywords[:3])
            return f"""关于{keyword_str}的补充信息：

根据查询"{query}"，以下信息可能对您有帮助：

1. 相关概念：{keyword_str}是检索优化中的重要技术
2. 应用场景：广泛应用于信息检索、问答系统、文档分析等领域
3. 技术优势：提升检索精度、改善用户体验、支持复杂查询
4. 发展趋势：结合深度学习和大语言模型不断演进

建议进一步查阅相关文献或技术文档获取详细信息。"""
        else:
            return f"""关于查询"{query}"的补充信息：

该查询涉及检索优化相关技术，可能包括：
- 重排序技术：提升检索结果的相关性排序
- 压缩技术：减少输入长度，提高处理效率
- 校正技术：评估和优化检索质量
- 多级优化：结合多种技术提升整体性能

建议提供更具体的查询以获取更精确的信息。"""
    
    def correct_retrieval_results(self, query: str, documents: List[Document], 
                                 evaluation_result: Dict[str, Any]) -> List[Document]:
        """校正检索结果"""
        print("校正检索结果...")
        
        quality_level = evaluation_result.get("quality_level", "poor")
        
        if quality_level == "good":
            print("  检索质量良好，无需校正")
            return documents
        
        print(f"  检索质量: {quality_level}，进行校正...")
        
        corrected_docs = documents.copy()
        
        # 1. 根据相关性分数重新排序
        if "relevance_score" in evaluation_result:
            relevance_score = evaluation_result["relevance_score"]
            if relevance_score < 0.5:
                print("  相关性较低，尝试重新排序...")
                # 简单按关键词匹配重新排序
                corrected_docs = self._reorder_by_keyword_match(query, corrected_docs)
        
        # 2. 补充缺失信息
        information_gaps = evaluation_result.get("information_gaps", [])
        if information_gaps:
            print(f"  补充{len(information_gaps)}个信息缺口...")
            supplementary_docs = self.supplement_external_knowledge(query, information_gaps)
            corrected_docs.extend(supplementary_docs)
        
        # 3. 过滤低质量文档
        if len(corrected_docs) > self.config.RERANK_TOP_K * 2:
            print("  文档数量过多，进行过滤...")
            corrected_docs = self._filter_low_quality_docs(corrected_docs)
        
        print(f"  校正完成: {len(documents)} -> {len(corrected_docs)} 个文档")
        return corrected_docs
    
    def _reorder_by_keyword_match(self, query: str, documents: List[Document]) -> List[Document]:
        """按关键词匹配重新排序"""
        query_words = set(query.lower().split())
        
        # 计算每个文档的匹配分数
        doc_scores = []
        for doc in documents:
            content = doc.page_content.lower()
            doc_words = set(content.split())
            
            # 计算匹配度
            intersection = len(query_words.intersection(doc_words))
            score = intersection / max(len(query_words), 1)
            
            doc_scores.append((doc, score))
        
        # 按分数排序
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 返回排序后的文档
        return [doc for doc, score in doc_scores]
    
    def _filter_low_quality_docs(self, documents: List[Document], max_docs: int = 10) -> List[Document]:
        """过滤低质量文档"""
        if len(documents) <= max_docs:
            return documents
        
        # 简单按长度过滤（假设较长的文档包含更多信息）
        sorted_docs = sorted(documents, key=lambda x: len(x.page_content), reverse=True)
        return sorted_docs[:max_docs]
    
    def correction_pipeline(self, query: str, documents: List[Document]) -> Tuple[List[Document], Dict[str, Any]]:
        """校正管道"""
        print("=" * 50)
        print("校正管道 (C-RAG)")
        print("=" * 50)
        
        if not documents:
            print("没有文档可校正")
            return [], {"error": "没有输入文档"}
        
        print(f"查询: {query}")
        print(f"输入文档数: {len(documents)}")
        
        # 1. 评估检索质量
        print("\n1. 评估检索质量...")
        evaluation_result = self.evaluate_retrieval_quality(query, documents)
        
        # 2. 校正检索结果
        print("\n2. 校正检索结果...")
        corrected_docs = self.correct_retrieval_results(query, documents, evaluation_result)
        
        # 3. 评估校正效果
        print("\n3. 评估校正效果...")
        if corrected_docs:
            post_correction_evaluation = self.evaluate_retrieval_quality(query, corrected_docs)
            
            # 计算改进程度
            original_score = evaluation_result.get("overall_score", 0.0)
            corrected_score = post_correction_evaluation.get("overall_score", 0.0)
            improvement = corrected_score - original_score
            
            print(f"  原始分数: {original_score:.3f}")
            print(f"  校正后分数: {corrected_score:.3f}")
            print(f"  改进程度: {improvement:+.3f}")
            
            if improvement > 0:
                print(f"  ✅ 校正有效，质量提升 {improvement:.3f}")
            elif improvement == 0:
                print(f"  ⚠️  校正无变化")
            else:
                print(f"  ❌ 校正后质量下降 {improvement:.3f}")
        else:
            post_correction_evaluation = {"error": "校正后无文档"}
            print("  校正后无文档")
        
        print(f"\n校正完成:")
        print(f"  原始文档数: {len(documents)}")
        print(f"  校正后文档数: {len(corrected_docs)}")
        
        return corrected_docs, {
            "original_evaluation": evaluation_result,
            "post_correction_evaluation": post_correction_evaluation,
            "improvement": improvement if corrected_docs else 0.0
        }
