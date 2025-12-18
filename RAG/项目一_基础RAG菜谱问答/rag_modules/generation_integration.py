from typing import List
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from config import Config

class GenerationIntegration:
    """生成集成模块"""
    
    def __init__(self):
        self.config = Config
        self.llm = self._load_llm()
        self.chain = self._create_chain()
    
    def _load_llm(self):
        """加载语言模型"""
        print(f"正在加载语言模型: {self.config.DEEPSEEK_MODEL}")
        
        llm = ChatOpenAI(
            model=self.config.DEEPSEEK_MODEL,
            openai_api_key=self.config.DEEPSEEK_API_KEY,
            openai_api_base=self.config.DEEPSEEK_API_BASE,
            temperature=self.config.TEMPERATURE,
            max_tokens=self.config.MAX_TOKENS
        )
        
        print("语言模型加载完成")
        return llm
    
    def _create_chain(self):
        """创建RAG链"""
        # 定义提示模板
        template = """你是一个专业的菜谱助手，请根据提供的菜谱信息回答用户的问题。
        
        相关菜谱信息：
        {context}
        
        用户问题：{question}
        
        请根据以上信息，用中文回答用户的问题。如果信息不足，请说明哪些信息缺失。
        
        回答："""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # 创建RAG链
        chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain
    
    def format_context(self, documents: List[Document]) -> str:
        """格式化检索到的上下文"""
        context_parts = []
        
        for i, doc in enumerate(documents):
            source = doc.metadata.get("source", "未知菜谱")
            content = doc.page_content
            
            context_parts.append(f"[文档 {i+1}] 来自: {source}")
            context_parts.append(content)
            context_parts.append("")  # 空行分隔
        
        return "\n".join(context_parts)
    
    def generate_answer(self, question: str, documents: List[Document]) -> str:
        """生成答案"""
        print(f"生成答案，问题: {question}")
        
        # 格式化上下文
        context = self.format_context(documents)
        
        # 生成答案
        answer = self.chain.invoke({
            "context": context,
            "question": question
        })
        
        print("答案生成完成")
        return answer
    
    def qa_pipeline(self, question: str, retriever) -> str:
        """完整的问答管道"""
        # 检索相关文档
        documents = retriever(question)
        
        # 生成答案
        answer = self.generate_answer(question, documents)
        
        return answer
