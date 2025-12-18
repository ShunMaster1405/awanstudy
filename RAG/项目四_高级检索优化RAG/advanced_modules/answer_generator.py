"""
答案生成器模块 - 高级检索优化版本
"""

import json
from typing import List, Dict, Any
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from config import Config

class AnswerGenerator:
    """答案生成器 - 支持高级优化技术"""
    
    def __init__(self):
        self.config = Config
        
        # 初始化LLM
        self.llm = self._initialize_llm()
        
        # 定义提示模板
        self.prompt_templates = self._create_prompt_templates()
    
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
            print(f"LLM初始化失败: {e}")
            print("使用备用配置...")
            
            # 备用配置
            llm = ChatOpenAI(
                model="deepseek-chat",
                openai_api_key="yourskindeepseek",
                openai_api_base="https://api.deepseek.com/v1",
                temperature=0.7,
                max_tokens=1000
            )
            return llm
    
    def _create_prompt_templates(self) -> Dict[str, ChatPromptTemplate]:
        """创建提示模板"""
        
        # 基础RAG提示
        basic_rag_template = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的AI助手，基于提供的文档内容回答问题。

请遵循以下规则：
1. 只基于提供的文档内容回答问题
2. 如果文档中没有相关信息，如实说明无法回答
3. 保持答案准确、简洁、有用
4. 引用相关的文档内容支持你的答案
5. 如果文档内容有矛盾，指出矛盾并给出最合理的解释

文档内容：
{document_content}

问题：{question}

请基于以上文档内容回答问题：""")
        ])
        
        # 高级优化RAG提示
        advanced_rag_template = ChatPromptTemplate.from_messages([
            ("system", """你是一个高级AI助手，基于经过优化的检索结果回答问题。

检索结果已经过以下优化处理：
1. 重排序：结果已按相关性重新排序
2. 压缩：文档内容已压缩，保留关键信息
3. 校正：检索质量已评估和优化（C-RAG）

优化统计：
{optimization_stats}

文档内容：
{document_content}

问题：{question}

请基于以上优化后的文档内容回答问题，并注意：
1. 优先使用相关性高的文档
2. 注意文档的压缩和校正标记
3. 如果文档中有补充内容，特别说明
4. 提供详细的推理过程

请回答：""")
        ])
        
        # 评估提示
        evaluation_template = ChatPromptTemplate.from_messages([
            ("system", """评估以下答案的质量：

问题：{question}
文档内容：{document_content}
生成的答案：{generated_answer}

请从以下维度评估答案质量：
1. 相关性：答案是否直接回答了问题
2. 准确性：答案是否基于文档内容，没有添加未提及的信息
3. 完整性：答案是否覆盖了问题的所有方面
4. 引用完整性：答案是否引用了相关的文档内容

请给出每个维度的评分（1-5分）和简要理由，然后给出总体评分（1-5分）。""")
        ])
        
        return {
            "basic": basic_rag_template,
            "advanced": advanced_rag_template,
            "evaluation": evaluation_template
        }
    
    def format_document_content(self, documents: List[Document]) -> str:
        """格式化文档内容"""
        formatted_content = []
        
        for i, doc in enumerate(documents):
            # 提取文档信息
            title = doc.metadata.get("title", f"文档 {i+1}")
            source = doc.metadata.get("source", "未知来源")
            
            # 检查优化标记
            optimization_marks = []
            if doc.metadata.get("reranking_method"):
                optimization_marks.append(f"重排序({doc.metadata['reranking_method']})")
            if doc.metadata.get("compression_method"):
                optimization_marks.append(f"压缩({doc.metadata['compression_method']})")
            if doc.metadata.get("correction_type"):
                optimization_marks.append(f"校正({doc.metadata['correction_type']})")
            if doc.metadata.get("supplementary", False):
                optimization_marks.append("补充")
            
            # 构建文档头
            doc_header = f"【文档 {i+1}: {title}】"
            if source:
                doc_header += f" ({source})"
            if optimization_marks:
                doc_header += f" [优化: {', '.join(optimization_marks)}]"
            
            # 添加文档内容
            doc_content = doc.page_content
            
            # 如果内容太长，进行截断
            max_length = 500
            if len(doc_content) > max_length:
                doc_content = doc_content[:max_length] + "..."
            
            formatted_content.append(f"{doc_header}\n{doc_content}")
        
        return "\n\n".join(formatted_content)
    
    def generate_optimization_stats(self, documents: List[Document]) -> str:
        """生成优化统计"""
        stats = {
            "total_documents": len(documents),
            "reranked_documents": 0,
            "compressed_documents": 0,
            "corrected_documents": 0,
            "supplementary_documents": 0
        }
        
        for doc in documents:
            if doc.metadata.get("reranking_method"):
                stats["reranked_documents"] += 1
            if doc.metadata.get("compression_method"):
                stats["compressed_documents"] += 1
            if doc.metadata.get("correction_type"):
                stats["corrected_documents"] += 1
            if doc.metadata.get("supplementary", False):
                stats["supplementary_documents"] += 1
        
        # 格式化统计信息
        stats_text = f"文档总数: {stats['total_documents']}\n"
        if stats['reranked_documents'] > 0:
            stats_text += f"重排序文档: {stats['reranked_documents']}\n"
        if stats['compressed_documents'] > 0:
            stats_text += f"压缩文档: {stats['compressed_documents']}\n"
        if stats['corrected_documents'] > 0:
            stats_text += f"校正文档: {stats['corrected_documents']}\n"
        if stats['supplementary_documents'] > 0:
            stats_text += f"补充文档: {stats['supplementary_documents']}\n"
        
        return stats_text
    
    def generate_basic_answer(self, question: str, documents: List[Document]) -> Dict[str, Any]:
        """生成基础答案"""
        print("生成基础答案...")
        
        if not documents:
            return {
                "answer": "抱歉，没有检索到相关文档，无法回答这个问题。",
                "documents_used": 0,
                "method": "basic",
                "error": "没有相关文档"
            }
        
        # 格式化文档内容
        document_content = self.format_document_content(documents)
        
        # 创建提示链
        chain = self.prompt_templates["basic"] | self.llm | StrOutputParser()
        
        try:
            # 生成答案
            answer = chain.invoke({
                "question": question,
                "document_content": document_content
            })
            
            return {
                "answer": answer,
                "documents_used": len(documents),
                "method": "basic",
                "document_content_preview": document_content[:500] + "..." if len(document_content) > 500 else document_content
            }
            
        except Exception as e:
            print(f"答案生成失败: {e}")
            return {
                "answer": f"抱歉，生成答案时出现错误: {str(e)}",
                "documents_used": len(documents),
                "method": "basic",
                "error": str(e)
            }
    
    def generate_advanced_answer(self, question: str, documents: List[Document], 
                               retrieval_stats: Dict[str, Any] = None) -> Dict[str, Any]:
        """生成高级答案（使用优化技术）"""
        print("生成高级答案...")
        
        if not documents:
            return {
                "answer": "抱歉，没有检索到相关文档，无法回答这个问题。",
                "documents_used": 0,
                "method": "advanced",
                "error": "没有相关文档"
            }
        
        # 格式化文档内容
        document_content = self.format_document_content(documents)
        
        # 生成优化统计
        optimization_stats = self.generate_optimization_stats(documents)
        
        # 添加检索统计（如果有）
        if retrieval_stats:
            optimization_stats += f"\n检索管道统计:\n"
            for stage in retrieval_stats.get("retrieval_pipeline", []):
                optimization_stats += f"  - {stage['stage']}: {stage['docs']} 个文档\n"
            
            if "correction_info" in retrieval_stats:
                correction_info = retrieval_stats["correction_info"]
                if "improvement" in correction_info:
                    improvement = correction_info["improvement"]
                    if improvement > 0:
                        optimization_stats += f"  校正改进: +{improvement:.3f}\n"
        
        # 创建提示链
        chain = self.prompt_templates["advanced"] | self.llm | StrOutputParser()
        
        try:
            # 生成答案
            answer = chain.invoke({
                "question": question,
                "document_content": document_content,
                "optimization_stats": optimization_stats
            })
            
            result = {
                "answer": answer,
                "documents_used": len(documents),
                "method": "advanced",
                "optimization_stats": optimization_stats,
                "document_content_preview": document_content[:500] + "..." if len(document_content) > 500 else document_content
            }
            
            # 添加检索统计
            if retrieval_stats:
                result["retrieval_stats"] = retrieval_stats
            
            return result
            
        except Exception as e:
            print(f"高级答案生成失败: {e}")
            return {
                "answer": f"抱歉，生成高级答案时出现错误: {str(e)}",
                "documents_used": len(documents),
                "method": "advanced",
                "error": str(e)
            }
    
    def evaluate_answer_quality(self, question: str, documents: List[Document], 
                              generated_answer: str) -> Dict[str, Any]:
        """评估答案质量"""
        print("评估答案质量...")
        
        if not documents or not generated_answer:
            return {
                "overall_score": 0.0,
                "dimensions": {},
                "evaluation": "无法评估：缺少文档或答案"
            }
        
        # 格式化文档内容
        document_content = self.format_document_content(documents)
        
        # 创建评估链
        chain = self.prompt_templates["evaluation"] | self.llm | StrOutputParser()
        
        try:
            # 生成评估
            evaluation_text = chain.invoke({
                "question": question,
                "document_content": document_content,
                "generated_answer": generated_answer
            })
            
            # 解析评估结果（简单解析）
            evaluation_result = {
                "evaluation_text": evaluation_text,
                "overall_score": self._extract_overall_score(evaluation_text),
                "dimensions": self._extract_dimension_scores(evaluation_text)
            }
            
            print(f"答案质量评估完成: 总体分数 {evaluation_result['overall_score']}/5")
            return evaluation_result
            
        except Exception as e:
            print(f"答案质量评估失败: {e}")
            return {
                "overall_score": 0.0,
                "dimensions": {},
                "evaluation": f"评估失败: {str(e)}",
                "error": str(e)
            }
    
    def _extract_overall_score(self, evaluation_text: str) -> float:
        """从评估文本中提取总体分数"""
        import re
        
        # 查找总体评分
        patterns = [
            r"总体评分[：:]\s*(\d+(?:\.\d+)?)/5",
            r"总体分数[：:]\s*(\d+(?:\.\d+)?)/5",
            r"overall score[：:]\s*(\d+(?:\.\d+)?)/5",
            r"总分[：:]\s*(\d+(?:\.\d+)?)/5"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, evaluation_text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        
        # 如果没有找到，尝试查找数字
        numbers = re.findall(r"\b(\d+(?:\.\d+)?)/5\b", evaluation_text)
        if numbers:
            try:
                return float(numbers[-1])  # 取最后一个分数
            except ValueError:
                pass
        
        return 3.0  # 默认分数
    
    def _extract_dimension_scores(self, evaluation_text: str) -> Dict[str, float]:
        """从评估文本中提取维度分数"""
        import re
        
        dimensions = {}
        
        # 常见维度
        dimension_patterns = {
            "相关性": r"相关性[：:]\s*(\d+(?:\.\d+)?)/5",
            "准确性": r"准确性[：:]\s*(\d+(?:\.\d+)?)/5",
            "完整性": r"完整性[：:]\s*(\d+(?:\.\d+)?)/5",
            "引用完整性": r"引用完整性[：:]\s*(\d+(?:\.\d+)?)/5",
            "relevance": r"relevance[：:]\s*(\d+(?:\.\d+)?)/5",
            "accuracy": r"accuracy[：:]\s*(\d+(?:\.\d+)?)/5",
            "completeness": r"completeness[：:]\s*(\d+(?:\.\d+)?)/5",
            "citation": r"citation[：:]\s*(\d+(?:\.\d+)?)/5"
        }
        
        for dimension, pattern in dimension_patterns.items():
            match = re.search(pattern, evaluation_text, re.IGNORECASE)
            if match:
                try:
                    dimensions[dimension] = float(match.group(1))
                except ValueError:
                    continue
        
        return dimensions
    
    def generate_pipeline(self, question: str, documents: List[Document], 
                         retrieval_stats: Dict[str, Any] = None) -> Dict[str, Any]:
        """生成管道"""
        print("=" * 50)
        print("答案生成管道")
        print("=" * 50)
        
        print(f"问题: {question}")
        print(f"输入文档数: {len(documents)}")
        
        if not documents:
            print("没有文档，无法生成答案")
            return {
                "success": False,
                "error": "没有输入文档",
                "question": question
            }
        
        # 生成高级答案
        print("\n1. 生成高级答案...")
        answer_result = self.generate_advanced_answer(question, documents, retrieval_stats)
        
        if "error" in answer_result:
            print(f"答案生成失败: {answer_result['error']}")
            return answer_result
        
        # 评估答案质量
        print("\n2. 评估答案质量...")
        evaluation_result = self.evaluate_answer_quality(
            question, 
            documents, 
            answer_result["answer"]
        )
        
        # 合并结果
        final_result = {
            "success": True,
            "question": question,
            "answer": answer_result["answer"],
            "documents_used": answer_result["documents_used"],
            "method": answer_result["method"],
            "evaluation": evaluation_result,
            "retrieval_stats": retrieval_stats
        }
        
        # 添加优化统计
        if "optimization_stats" in answer_result:
            final_result["optimization_stats"] = answer_result["optimization_stats"]
        
        print(f"\n答案生成完成:")
        print(f"  文档使用数: {answer_result['documents_used']}")
        print(f"  答案质量分数: {evaluation_result.get('overall_score', 0.0)}/5")
        
        return final_result
