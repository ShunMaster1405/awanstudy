"""
答案生成器模块
"""

import os
from typing import List, Dict, Any
from openai import OpenAI
from config import Config

class AnswerGenerator:
    """答案生成器 - 使用DeepSeek生成答案"""
    
    def __init__(self):
        self.config = Config
        
        # 初始化DeepSeek客户端
        print(f"初始化DeepSeek LLM: {self.config.DEEPSEEK_MODEL}")
        self.client = OpenAI(
            api_key=self.config.DEEPSEEK_API_KEY,
            base_url=self.config.DEEPSEEK_API_BASE
        )
        
        # 系统提示词
        self.system_prompt = """你是一个专业的AI助手，基于提供的文档内容回答问题。
        
        请遵循以下规则：
        1. 只基于提供的文档内容回答问题，不要添加外部知识
        2. 如果文档中没有相关信息，请明确说明"根据提供的文档，无法回答这个问题"
        3. 回答要简洁、准确、有条理
        4. 可以引用文档中的关键信息
        5. 如果文档内容有矛盾，指出矛盾之处
        
        文档内容如下："""
    
    def generate_answer(self, question: str, retrieval_results: List[Dict[str, Any]]) -> str:
        """生成答案"""
        if not retrieval_results:
            return "根据提供的文档，无法回答这个问题。"
        
        # 准备上下文
        context = self._prepare_context(retrieval_results)
        
        # 构建提示词
        prompt = self._build_prompt(question, context)
        
        try:
            # 调用DeepSeek API
            response = self.client.chat.completions.create(
                model=self.config.DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": self.system_prompt + context},
                    {"role": "user", "content": question}
                ],
                max_tokens=self.config.MAX_TOKENS,
                temperature=self.config.TEMPERATURE
            )
            
            answer = response.choices[0].message.content
            
            # 添加检索信息
            answer_with_info = self._add_retrieval_info(answer, retrieval_results)
            
            return answer_with_info
            
        except Exception as e:
            print(f"生成答案失败: {e}")
            # 返回基于检索结果的简单答案
            return self._generate_fallback_answer(question, retrieval_results)
    
    def _prepare_context(self, retrieval_results: List[Dict[str, Any]]) -> str:
        """准备上下文"""
        context_parts = []
        
        for i, result in enumerate(retrieval_results[:3]):  # 使用前3个结果
            content = result["content"]
            metadata = result["metadata"]
            score = result.get("score", 0.0)
            retrieval_type = result.get("type", "unknown")
            
            # 格式化上下文
            context_part = f"""
【文档 {i+1} - {retrieval_type}检索，相关性分数: {score:.3f}】
来源: {metadata.get('title', '未知')}
内容: {content[:500]}...  # 限制长度
"""
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _build_prompt(self, question: str, context: str) -> str:
        """构建提示词"""
        prompt = f"""基于以下文档内容回答问题：

{context}

问题: {question}

请基于上述文档内容提供准确、简洁的回答。如果文档中没有相关信息，请明确说明。"""
        
        return prompt
    
    def _add_retrieval_info(self, answer: str, retrieval_results: List[Dict[str, Any]]) -> str:
        """添加检索信息"""
        if not retrieval_results:
            return answer
        
        # 统计信息
        total_results = len(retrieval_results)
        hybrid_results = [r for r in retrieval_results if r.get("type") == "hybrid"]
        dense_results = [r for r in retrieval_results if r.get("type") == "dense"]
        sparse_results = [r for r in retrieval_results if r.get("type") == "sparse"]
        
        # 最高分数
        top_score = max([r.get("score", 0.0) for r in retrieval_results]) if retrieval_results else 0.0
        
        # 添加信息
        info = f"""

【检索统计】
- 总检索结果: {total_results} 个文档
- 混合检索结果: {len(hybrid_results)} 个
- 密集检索结果: {len(dense_results)} 个  
- 稀疏检索结果: {len(sparse_results)} 个
- 最高相关性分数: {top_score:.3f}

注：答案基于检索到的文档内容生成，如果信息不完整或存在矛盾，可能是由于文档覆盖不足。"""
        
        return answer + info
    
    def _generate_fallback_answer(self, question: str, retrieval_results: List[Dict[str, Any]]) -> str:
        """生成备用答案（当API调用失败时）"""
        if not retrieval_results:
            return "无法回答该问题，因为未找到相关文档。"
        
        # 提取所有内容
        contents = [r["content"] for r in retrieval_results[:2]]  # 使用前2个结果
        
        # 简单关键词匹配
        question_lower = question.lower()
        relevant_parts = []
        
        for content in contents:
            content_lower = content.lower()
            
            # 检查问题关键词是否在内容中
            keywords = question_lower.split()
            matched_keywords = [kw for kw in keywords if kw in content_lower and len(kw) > 2]
            
            if matched_keywords:
                # 提取包含关键词的句子
                sentences = content.split('。')
                relevant_sentences = []
                
                for sentence in sentences:
                    sentence_lower = sentence.lower()
                    if any(kw in sentence_lower for kw in matched_keywords):
                        relevant_sentences.append(sentence.strip())
                
                if relevant_sentences:
                    relevant_parts.extend(relevant_sentences[:2])  # 每个文档最多2个句子
        
        if relevant_parts:
            # 构建简单答案
            answer = f"基于检索到的文档，以下信息可能与您的问题相关：\n\n"
            for i, part in enumerate(relevant_parts[:3], 1):
                answer += f"{i}. {part}\n"
            
            answer += f"\n注：这是基于关键词匹配的简单回答，可能不完整。"
            return answer
        else:
            return "在检索到的文档中未找到与问题直接相关的内容。"
    
    def evaluate_answer_quality(self, question: str, answer: str, retrieval_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """评估答案质量"""
        if not retrieval_results:
            return {
                "has_context": False,
                "relevance_score": 0.0,
                "completeness_score": 0.0,
                "overall_quality": "低"
            }
        
        # 简单评估逻辑
        answer_lower = answer.lower()
        question_lower = question.lower()
        
        # 1. 检查是否包含"无法回答"等短语
        if any(phrase in answer_lower for phrase in ["无法回答", "没有找到", "未找到", "不知道", "不清楚"]):
            has_answer = False
        else:
            has_answer = True
        
        # 2. 计算相关性（答案中是否包含问题关键词）
        question_keywords = [kw for kw in question_lower.split() if len(kw) > 2]
        matched_keywords = [kw for kw in question_keywords if kw in answer_lower]
        keyword_coverage = len(matched_keywords) / len(question_keywords) if question_keywords else 0.0
        
        # 3. 检查是否引用了检索结果
        has_citations = "【检索统计】" in answer or "来源:" in answer
        
        # 4. 答案长度（适中的长度更好）
        answer_length = len(answer)
        if answer_length < 50:
            length_score = 0.3
        elif answer_length < 200:
            length_score = 0.7
        elif answer_length < 500:
            length_score = 0.9
        else:
            length_score = 0.6
        
        # 综合评分
        relevance_score = keyword_coverage * 0.6 + (1.0 if has_answer else 0.0) * 0.4
        completeness_score = length_score * 0.5 + (1.0 if has_citations else 0.0) * 0.5
        overall_score = (relevance_score + completeness_score) / 2
        
        # 质量等级
        if overall_score >= 0.8:
            quality = "高"
        elif overall_score >= 0.6:
            quality = "中"
        elif overall_score >= 0.4:
            quality = "低"
        else:
            quality = "很差"
        
        return {
            "has_context": True,
            "has_answer": has_answer,
            "keyword_coverage": keyword_coverage,
            "has_citations": has_citations,
            "answer_length": answer_length,
            "relevance_score": relevance_score,
            "completeness_score": completeness_score,
            "overall_score": overall_score,
            "overall_quality": quality
        }
