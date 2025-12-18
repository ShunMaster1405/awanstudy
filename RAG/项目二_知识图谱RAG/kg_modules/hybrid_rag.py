"""
混合RAG模块
"""

from typing import List, Callable, Dict, Any
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from config import Config

class HybridRAG:
    """混合RAG模块"""
    
    def __init__(self, vector_retriever: Callable, graph_retriever: Callable):
        self.vector_retriever = vector_retriever
        self.graph_retriever = graph_retriever
        self.llm = self._init_llm()
        self.prompt = self._create_prompt()
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    def _init_llm(self):
        """初始化LLM"""
        print("初始化DeepSeek LLM...")
        
        llm = ChatOpenAI(
            api_key=Config.DEEPSEEK_API_KEY,
            base_url=Config.DEEPSEEK_API_BASE,
            model=Config.DEEPSEEK_MODEL,
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS
        )
        
        return llm
    
    def _create_prompt(self):
        """创建提示模板"""
        template = """你是一个基于知识图谱的智能助手。请根据以下信息回答问题：

## 向量检索结果（基于语义相似度）：
{vector_context}

## 图检索结果（基于知识图谱）：
{graph_context}

## 用户问题：
{question}

## 回答要求：
1. 结合两种检索结果提供准确答案
2. 如果信息不足，请说明哪些方面信息不足
3. 保持回答简洁明了
4. 如果可能，展示知识图谱中的关系

请用中文回答："""

        return ChatPromptTemplate.from_template(template)
    
    def retrieve_context(self, question: str) -> Dict[str, Any]:
        """检索上下文"""
        print("检索上下文...")
        
        # 向量检索
        print("  - 执行向量检索...")
        vector_results = self.vector_retriever(question)
        vector_context = self._format_vector_results(vector_results)
        
        # 图检索
        print("  - 执行图检索...")
        graph_results = self.graph_retriever(question)
        graph_context = self._format_graph_results(graph_results)
        
        return {
            "vector_context": vector_context,
            "graph_context": graph_context,
            "vector_results": vector_results,
            "graph_results": graph_results
        }
    
    def _format_vector_results(self, results: List[Document]) -> str:
        """格式化向量检索结果"""
        if not results:
            return "没有找到相关的文档内容。"
        
        formatted = ""
        for i, doc in enumerate(results, 1):
            content = doc.page_content
            # 截断过长的内容
            if len(content) > 300:
                content = content[:300] + "..."
            
            source = doc.metadata.get("source", "未知")
            title = doc.metadata.get("title", "无标题")
            
            formatted += f"{i}. 【{title}】{content}\n   来源: {source}\n\n"
        
        return formatted.strip()
    
    def _format_graph_results(self, results: List[Document]) -> str:
        """格式化图检索结果"""
        if not results:
            return "没有找到相关的知识图谱信息。"
        
        formatted = ""
        for i, doc in enumerate(results, 1):
            content = doc.page_content
            source = doc.metadata.get("source", "未知")
            doc_type = doc.metadata.get("type", "未知")
            
            formatted += f"{i}. 【{doc_type}】{content}\n   来源: {source}\n\n"
        
        return formatted.strip()
    
    def answer_question(self, question: str) -> str:
        """回答问题"""
        print(f"\n处理问题: {question}")
        
        try:
            # 检索上下文
            context = self.retrieve_context(question)
            
            # 生成答案
            print("生成答案...")
            answer = self.chain.invoke({
                "vector_context": context["vector_context"],
                "graph_context": context["graph_context"],
                "question": question
            })
            
            # 添加检索统计信息
            stats = f"\n\n【检索统计】\n"
            stats += f"- 向量检索结果: {len(context['vector_results'])} 个文档\n"
            stats += f"- 图检索结果: {len(context['graph_results'])} 个知识片段\n"
            
            answer += stats
            
            return answer
            
        except Exception as e:
            error_msg = f"回答问题时出错: {str(e)}"
            print(error_msg)
            return f"抱歉，处理问题时出现错误：{str(e)}"
    
    def analyze_retrieval(self, question: str) -> Dict[str, Any]:
        """分析检索结果"""
        context = self.retrieve_context(question)
        
        return {
            "question": question,
            "vector_results_count": len(context["vector_results"]),
            "graph_results_count": len(context["graph_results"]),
            "vector_context_preview": context["vector_context"][:200] + "..." if context["vector_context"] else "无",
            "graph_context_preview": context["graph_context"][:200] + "..." if context["graph_context"] else "无",
            "retrieval_summary": self._generate_retrieval_summary(context)
        }
    
    def _generate_retrieval_summary(self, context: Dict[str, Any]) -> str:
        """生成检索摘要"""
        vector_count = len(context["vector_results"])
        graph_count = len(context["graph_results"])
        
        if vector_count > 0 and graph_count > 0:
            return f"成功检索到 {vector_count} 个相关文档和 {graph_count} 个知识图谱片段。"
        elif vector_count > 0:
            return f"成功检索到 {vector_count} 个相关文档，但未找到相关的知识图谱信息。"
        elif graph_count > 0:
            return f"成功检索到 {graph_count} 个知识图谱片段，但未找到相关的文档内容。"
        else:
            return "未找到相关的检索结果。"
