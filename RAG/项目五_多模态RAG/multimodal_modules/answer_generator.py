"""
多模态答案生成器
"""

import os
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from config import Config

class MultimodalAnswerGenerator:
    """多模态答案生成器 - 支持多模态上下文生成"""
    
    def __init__(self):
        self.config = Config
        self.llm = self._initialize_llm()
        
        print("初始化多模态答案生成器...")
    
    def _initialize_llm(self):
        """初始化LLM"""
        print("初始化DeepSeek LLM...")
        
        try:
            llm = ChatOpenAI(
                model=self.config.DEEPSEEK_MODEL,
                openai_api_key=self.config.DEEPSEEK_API_KEY,
                openai_api_base=self.config.DEEPSEEK_API_BASE,
                temperature=self.config.TEMPERATURE,
                max_tokens=self.config.MAX_TOKENS
            )
            print(f"  使用模型: {self.config.DEEPSEEK_MODEL}")
            return llm
        except Exception as e:
            print(f"❌ 初始化LLM失败: {e}")
            return None
    
    def generate_pipeline(self, query: str, documents: List[Document], 
                         retrieval_stats: Dict[str, Any], image_path: Optional[str] = None) -> Dict[str, Any]:
        """多模态答案生成管道"""
        print("=" * 50)
        print("多模态答案生成管道")
        print("=" * 50)
        
        if not self.llm:
            print("❌ LLM未初始化")
            return {"success": False, "error": "LLM未初始化"}
        
        if not documents:
            print("❌ 没有文档可用于生成答案")
            return {"success": False, "error": "没有检索到相关文档"}
        
        print(f"问题: {query}")
        print(f"输入文档数: {len(documents)}")
        if image_path:
            print(f"图像: {image_path}")
        
        # 1. 准备多模态上下文
        print("\n1. 准备多模态上下文...")
        context = self._prepare_multimodal_context(documents, retrieval_stats, image_path)
        
        # 2. 生成多模态答案
        print("2. 生成多模态答案...")
        answer_result = self._generate_multimodal_answer(query, context, image_path)
        
        if not answer_result.get("success", False):
            print("❌ 答案生成失败")
            return answer_result
        
        # 3. 评估答案质量
        print("3. 评估答案质量...")
        evaluation = self._evaluate_answer_quality(query, answer_result["answer"], documents, image_path)
        
        # 4. 整合结果
        final_result = {
            "success": True,
            "question": query,
            "answer": answer_result["answer"],
            "documents_used": len(documents),
            "evaluation": evaluation,
            "retrieval_stats": retrieval_stats,
            "has_image_context": bool(image_path)
        }
        
        print("✅ 答案生成完成")
        return final_result
    
    def _prepare_multimodal_context(self, documents: List[Document], 
                                   retrieval_stats: Dict[str, Any], 
                                   image_path: Optional[str] = None) -> str:
        """准备多模态上下文"""
        context_parts = []
        
        # 添加检索统计信息
        stats_info = self._format_retrieval_stats(retrieval_stats)
        context_parts.append(f"检索统计信息:\n{stats_info}")
        
        # 添加文档内容
        context_parts.append("\n相关文档内容:")
        
        for i, doc in enumerate(documents, 1):
            modality = doc.metadata.get("modality", "text")
            source = doc.metadata.get("source", "未知")
            title = doc.metadata.get("title", "无标题")
            
            # 格式化文档内容
            doc_content = doc.page_content.strip()
            if len(doc_content) > 300:
                doc_content = doc_content[:300] + "..."
            
            doc_info = f"\n文档{i} [{modality}]: {title} ({source})\n内容: {doc_content}"
            context_parts.append(doc_info)
        
        # 添加图像信息（如果有）
        if image_path:
            image_info = f"\n图像上下文: 用户提供了图像文件 '{os.path.basename(image_path)}'，请结合图像内容回答问题。"
            context_parts.append(image_info)
        
        # 添加模态分布信息
        modality_dist = retrieval_stats.get("stats", {}).get("modality_distribution", {})
        if modality_dist:
            modality_info = "\n检索结果模态分布:"
            for modality, count in modality_dist.items():
                modality_info += f"\n  - {modality}: {count} 个文档"
            context_parts.append(modality_info)
        
        return "\n".join(context_parts)
    
    def _format_retrieval_stats(self, retrieval_stats: Dict[str, Any]) -> str:
        """格式化检索统计信息"""
        stats = retrieval_stats.get("stats", {})
        
        formatted_stats = []
        formatted_stats.append(f"• 检索文档数: {stats.get('total_docs', 0)}")
        formatted_stats.append(f"• 平均相似度分数: {stats.get('avg_score', 0):.3f}")
        formatted_stats.append(f"• 查询相关性: {stats.get('query_relevance', 0):.2f}")
        formatted_stats.append(f"• 查询类型: {retrieval_stats.get('query_type', 'unknown')}")
        
        return "\n".join(formatted_stats)
    
    def _generate_multimodal_answer(self, query: str, context: str, 
                                   image_path: Optional[str] = None) -> Dict[str, Any]:
        """生成多模态答案"""
        try:
            # 构建多模态提示
            prompt = self._build_multimodal_prompt(query, context, image_path)
            
            # 调用LLM生成答案
            print("  调用LLM生成答案...")
            response = self.llm.invoke(prompt)
            
            answer = response.content.strip()
            
            return {
                "success": True,
                "answer": answer,
                "raw_response": response
            }
            
        except Exception as e:
            print(f"❌ 生成答案失败: {e}")
            return {"success": False, "error": str(e)}
    
    def _build_multimodal_prompt(self, query: str, context: str, 
                                image_path: Optional[str] = None) -> List:
        """构建多模态提示"""
        # 系统提示
        system_prompt = """你是一个多模态RAG系统的答案生成器。你的任务是基于提供的多模态上下文（包括文本和图像信息）生成准确、全面的答案。

请遵循以下原则：
1. 基于提供的上下文生成答案，不要编造信息
2. 如果上下文包含图像信息，请考虑图像内容
3. 如果上下文不足，请明确指出信息有限
4. 保持答案的专业性和准确性
5. 使用清晰、有条理的结构组织答案

请生成高质量的答案。"""
        
        # 用户提示
        user_prompt = f"""问题: {query}

多模态上下文:
{context}

请基于以上多模态上下文，生成一个全面、准确的答案。"""

        # 如果有图像路径，添加图像提示
        if image_path:
            user_prompt += f"\n\n注意：用户提供了图像文件 '{os.path.basename(image_path)}'，请结合图像内容回答问题。"
        
        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
    
    def _evaluate_answer_quality(self, query: str, answer: str, 
                                documents: List[Document], 
                                image_path: Optional[str] = None) -> Dict[str, Any]:
        """评估答案质量"""
        print("  评估答案质量...")
        
        try:
            # 构建评估提示
            evaluation_prompt = self._build_evaluation_prompt(query, answer, documents, image_path)
            
            # 调用LLM进行评估
            response = self.llm.invoke(evaluation_prompt)
            evaluation_text = response.content.strip()
            
            # 解析评估结果（简化版）
            evaluation = self._parse_evaluation_result(evaluation_text)
            
            return evaluation
            
        except Exception as e:
            print(f"❌ 评估答案质量失败: {e}")
            return {
                "overall_score": 3.0,
                "dimensions": {
                    "准确性": 3.0,
                    "完整性": 3.0,
                    "相关性": 3.0,
                    "清晰度": 3.0
                },
                "feedback": "评估失败，使用默认评分"
            }
    
    def _build_evaluation_prompt(self, query: str, answer: str, 
                                documents: List[Document], 
                                image_path: Optional[str] = None) -> List:
        """构建评估提示"""
        # 准备文档摘要
        doc_summary = ""
        for i, doc in enumerate(documents[:3], 1):  # 只使用前3个文档
            modality = doc.metadata.get("modality", "text")
            content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            doc_summary += f"\n文档{i} [{modality}]: {content_preview}"
        
        system_prompt = """你是一个答案质量评估器。请根据以下标准评估答案质量：
1. 准确性（1-5分）：答案是否基于提供的上下文，是否准确无误
2. 完整性（1-5分）：答案是否全面回答了问题，是否覆盖了所有重要方面
3. 相关性（1-5分）：答案是否与问题高度相关，是否解决了核心问题
4. 清晰度（1-5分）：答案是否清晰易懂，结构是否合理

请给出每个维度的评分（1-5分，5分为最高），并计算总体评分（四个维度的平均值）。
最后提供简短的反馈意见。"""

        user_prompt = f"""问题: {query}

参考文档摘要:{doc_summary}

生成的答案:
{answer}

{'注意：这个答案应该结合了图像上下文。' if image_path else ''}

请按照以下格式输出评估结果：
准确性: [分数]/5
完整性: [分数]/5
相关性: [分数]/5
清晰度: [分数]/5
总体评分: [平均分数]/5
反馈: [简短的反馈意见]"""

        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
    
    def _parse_evaluation_result(self, evaluation_text: str) -> Dict[str, Any]:
        """解析评估结果"""
        # 简化版解析
        lines = evaluation_text.strip().split('\n')
        
        dimensions = {}
        overall_score = 3.0  # 默认值
        
        for line in lines:
            line = line.strip()
            if '准确性:' in line:
                dimensions['准确性'] = self._extract_score(line)
            elif '完整性:' in line:
                dimensions['完整性'] = self._extract_score(line)
            elif '相关性:' in line:
                dimensions['相关性'] = self._extract_score(line)
            elif '清晰度:' in line:
                dimensions['清晰度'] = self._extract_score(line)
            elif '总体评分:' in line:
                overall_score = self._extract_score(line)
        
        # 如果解析失败，使用默认值
        if not dimensions:
            dimensions = {
                "准确性": 3.0,
                "完整性": 3.0,
                "相关性": 3.0,
                "清晰度": 3.0
            }
        
        # 计算总体评分（如果未解析到）
        if overall_score == 3.0 and dimensions:
            overall_score = sum(dimensions.values()) / len(dimensions)
        
        return {
            "overall_score": overall_score,
            "dimensions": dimensions,
            "raw_evaluation": evaluation_text
        }
    
    def _extract_score(self, text: str) -> float:
        """从文本中提取分数"""
        try:
            # 查找数字
            import re
            match = re.search(r'(\d+(\.\d+)?)/5', text)
            if match:
                return float(match.group(1))
            
            # 如果没有找到/5格式，尝试其他格式
            match = re.search(r'(\d+(\.\d+)?)', text)
            if match:
                score = float(match.group(1))
                # 确保分数在1-5范围内
                return max(1.0, min(5.0, score))
        except:
            pass
        
        return 3.0  # 默认值
    
    def multimodal_generation_demo(self):
        """多模态生成演示"""
        print("\n" + "=" * 50)
        print("多模态生成演示")
        print("=" * 50)
        
        demo_queries = [
            {
                "query": "请描述多模态学习的主要优势",
                "description": "多模态学习优势分析"
            },
            {
                "query": "图像识别技术在哪些领域有应用？",
                "description": "图像识别应用场景"
            },
            {
                "query": "如何实现文本和图像的联合表示？",
                "description": "跨模态表示学习"
            }
        ]
        
        for i, demo in enumerate(demo_queries, 1):
            print(f"\n演示 {i}: {demo['description']}")
            print(f"查询: {demo['query']}")
            
            # 创建模拟文档
            mock_documents = [
                Document(
                    page_content="多模态学习结合了文本、图像、音频等多种信息源，能够提供更全面的理解和更准确的预测。",
                    metadata={"modality": "text", "source": "text_1.txt", "title": "多模态学习概述"}
                ),
                Document(
                    page_content="图像识别技术已广泛应用于安防监控、医疗诊断、自动驾驶、工业检测等领域。",
                    metadata={"modality": "text", "source": "text_2.txt", "title": "图像识别应用"}
                )
            ]
            
            mock_stats = {
                "success": True,
                "retrieved_docs": [],
                "final_docs": 2,
                "stats": {
                    "total_docs": 2,
                    "avg_score": 0.85,
                    "modality_distribution": {"text": 2},
                    "query_relevance": 0.8
                },
                "query_type": "text_only"
            }
            
            # 生成答案
            result = self.generate_pipeline(demo["query"], mock_documents, mock_stats)
            
            if result.get("success", False):
                print(f"答案预览: {result['answer'][:150]}...")
                eval_score = result['evaluation'].get('overall_score', 0)
                print(f"答案质量评分: {eval_score}/5")
            else:
                print("❌ 生成失败")
            
            print("-" * 40)
