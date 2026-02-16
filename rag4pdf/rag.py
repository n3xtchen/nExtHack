import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import tenacity
from google import genai
from google.genai import errors

# 配置日志
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def is_retryable_error(exception):
    """
    判断异常是否应该重试：
    1. 包含 429 (速率限制) 或 503 (服务不可用) 的错误信息
    2. SDK 明确定义的客户端/服务器错误
    3. 解析异常 (有时候重试可能得到正确的格式)
    """
    msg = str(exception).lower()
    if "429" in msg or "503" in msg or "quota" in msg or "rate limit" in msg:
        return True
    if isinstance(exception, (errors.ClientError, errors.ServerError)):
        return True
    # 如果是 JSON 解析错误，也可以尝试重试，因为模型输出具有随机性
    if isinstance(exception, (json.JSONDecodeError, ValueError)):
        return True
    return False


def robust_json_parse(text: str):
    """
    鲁棒地解析 LLM 返回的 JSON 字符串，处理常见的 Markdown 标记。
    """
    if not text:
        raise ValueError("Empty response from LLM")

    # 尝试直接解析
    text_clean = text.strip()
    try:
        return json.loads(text_clean)
    except json.JSONDecodeError:
        pass

    # 尝试提取 ```json ... ``` 中的内容
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text_clean)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 如果还是不行，尝试寻找第一个 { 和最后一个 }
    start = text_clean.find("{")
    end = text_clean.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(text_clean[start : end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Failed to parse JSON from LLM output: {text[:100]}...")



class BaseRetriever:
    """
    Base class for retrievers.
    Subclasses should implement the fit and get_top_k methods.
    """

    def __init__(self):
        self.documents = []

    def fit(self, documents: List[str]):
        """Store the documents"""
        self.documents = documents

    def get_top_k(self, query: str, k: int = 3) -> List[tuple]:
        """Retrieve top-k most relevant documents for the query."""
        raise NotImplementedError("Subclasses should implement this method.")


class SimpleKeywordRetriever(BaseRetriever):
    """Ultra-simple keyword matching retriever"""

    def __init__(self, tokenizer: str = "split"):
        super().__init__()
        self.tokenizer = tokenizer

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text according to selected tokenizer"""
        if self.tokenizer == "jieba":
            try:
                import jieba

                return list(jieba.cut(text.lower()))
            except ImportError:
                import warnings

                warnings.warn("jieba not installed, falling back to split tokenizer")
                return text.lower().split()
        return text.lower().split()

    def _count_keyword_matches(self, query: str, document: str) -> int:
        """Count how many query words appear in the document"""
        query_words = self._tokenize(query)
        document_words = self._tokenize(document)
        matches = 0
        for word in query_words:
            if word in document_words:
                matches += 1
        return matches

    def get_top_k(self, query: str, k: int = 3) -> List[tuple]:
        """Get top k documents by keyword match count"""
        scores = []

        for i, doc in enumerate(self.documents):
            match_count = self._count_keyword_matches(query, doc)
            scores.append((i, match_count))

        # Sort by match count (descending)
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:k]


class SimpleRAG:
    """
    Simple RAG system that:
    1. accepts a llm client
    2. uses simple keyword matching to retrieve relevant documents
    3. uses the llm client to generate a response based on the retrieved documents when a query is made
    """

    def __init__(
        self,
        llm_client,
        retriever: Optional[BaseRetriever] = None,
        system_prompt: Optional[str] = None,
        model_name: str = "gemini-2.5-pro",
    ):
        """
        Initialize RAG system

        Args:
            llm_client: LLM client with a generate() method
            retriever: Document retriever (defaults to SimpleKeywordRetriever)
            system_prompt: System prompt template for generation
            model_name: Default model to use for generation
        """
        self.llm_client = llm_client
        self.retriever = retriever or SimpleKeywordRetriever()
        self.model_name = model_name
        self.system_prompt = (
            system_prompt
            or """基于提供的文档回答以下问题：
                                问题: {query}
                                文档:
                                {context}
                                回答:
                            """
        )
        self.documents = []
        self.is_fitted = False

    def add_documents(self, documents: List[str]):
        """Add documents to the knowledge base"""
        self.documents.extend(documents)
        # Refit retriever with all documents
        self.retriever.fit(self.documents)
        self.is_fitted = True

    def set_documents(self, documents: List[str]):
        """Set documents (replacing any existing ones)"""
        self.documents = documents
        self.retriever.fit(self.documents)
        self.is_fitted = True

    def retrieve_documents(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most relevant documents for the query

        Args:
            query: Search query
            top_k: Number of documents to retrieve

        Returns:
            List of dictionaries containing document info
        """
        if not self.is_fitted:
            raise ValueError(
                "No documents have been added. Call add_documents() or set_documents() first."
            )

        top_docs = self.retriever.get_top_k(query, k=top_k)

        retrieved_docs = []
        for idx, score in top_docs:
            if score > 0:  # Only include documents with positive similarity scores
                retrieved_docs.append(
                    {
                        "content": self.documents[idx],
                        "similarity_score": score,
                        "document_id": idx,
                    }
                )

        return retrieved_docs

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=2, max=30),
        stop=tenacity.stop_after_attempt(5),
        retry=tenacity.retry_if_exception(is_retryable_error),
        before_sleep=tenacity.before_sleep_log(logger, logging.INFO),
        reraise=True,
    )
    def generate_response(
        self, query: str, top_k: int = 3, model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate response to query using retrieved documents

        Args:
            query: User query
            top_k: Number of documents to retrieve
            model_name: Optional model to use for this specific request

        Returns:
            Dictionary containing 'answer' and 'retrieved_docs'
        """
        if not self.is_fitted:
            raise ValueError(
                "No documents have been added. Call add_documents() or set_documents() first."
            )

        # Retrieve relevant documents
        retrieved_docs = self.retrieve_documents(query, top_k)

        if not retrieved_docs:
            return {
                "answer": "I couldn't find any relevant documents to answer your question.",
                "retrieved_docs": []
            }

        # Build context from retrieved documents
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(f"Document {i}:\n{doc['content']}")

        context = "\n\n".join(context_parts)

        # Generate response using LLM client
        prompt = self.system_prompt.format(query=query, context=context)

        try:
            response = self.llm_client.models.generate_content(
                model=model_name or self.model_name, contents=prompt
            )
            return {
                "answer": response.text.strip(),
                "retrieved_docs": retrieved_docs
            }

        except Exception as e:
            if is_retryable_error(e):
                raise e
            return {
                "answer": f"Error generating response: {str(e)}",
                "retrieved_docs": retrieved_docs
            }

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=2, max=30),
        stop=tenacity.stop_after_attempt(5),
        retry=tenacity.retry_if_exception(is_retryable_error),
        reraise=True,
    )
    async def agenerate_response(
        self, query: str, top_k: int = 3, model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Async version of generate_response
        """
        if not self.is_fitted:
            raise ValueError(
                "No documents have been added. Call add_documents() or set_documents() first."
            )

        # Retrieve relevant documents
        retrieved_docs = self.retrieve_documents(query, top_k)

        if not retrieved_docs:
            return {
                "answer": "I couldn't find any relevant documents to answer your question.",
                "retrieved_docs": []
            }

        # Build context from retrieved documents
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(f"Document {i}:\n{doc['content']}")

        context = "\n\n".join(context_parts)

        # Generate response using LLM client
        prompt = self.system_prompt.format(query=query, context=context)

        try:
            response = await self.llm_client.aio.models.generate_content(
                model=model_name or self.model_name, contents=prompt
            )
            return {
                "answer": response.text.strip(),
                "retrieved_docs": retrieved_docs
            }

        except Exception as e:
            if is_retryable_error(e):
                raise e
            return {
                "answer": f"Error generating response: {str(e)}",
                "retrieved_docs": retrieved_docs
            }

    def query(
        self,
        question: str,
        top_k: int = 3,
        run_id: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve documents and generate response

        Args:
            question: User question
            top_k: Number of documents to retrieve
            run_id: Optional run ID for tracing (auto-generated if not provided)
            model_name: Optional model to use for this specific request

        Returns:
            Dictionary containing response, retrieved_docs and run_id
        """
        # Generate run_id if not provided
        if run_id is None:
            run_id = (
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(question) % 10000:04d}"
            )

        try:
            result = self.generate_response(question, top_k, model_name=model_name)
            return {
                "answer": result["answer"],
                "retrieved_docs": result["retrieved_docs"],
                "run_id": run_id
            }

        except Exception as e:
            # Return error result
            return {
                "answer": f"Error processing query: {str(e)}",
                "retrieved_docs": [],
                "run_id": run_id,
            }

    async def aquery(
        self,
        question: str,
        top_k: int = 3,
        run_id: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Async version of query
        """
        # Generate run_id if not provided
        if run_id is None:
            run_id = (
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(question) % 10000:04d}"
            )

        try:
            result = await self.agenerate_response(
                question, top_k, model_name=model_name
            )
            return {
                "answer": result["answer"],
                "retrieved_docs": result["retrieved_docs"],
                "run_id": run_id
            }

        except Exception as e:
            # Return error result
            return {
                "answer": f"Error processing query: {str(e)}",
                "retrieved_docs": [],
                "run_id": run_id,
            }



class SimpleAgent:
    def __init__(self, client, system_prompt: str = None):
        self.client = client
        self.system_prompt = (
            system_prompt
            or """回答如下问题：
problem：{user_input}
answer：

请使用中文回答 (Please respond in Chinese).


输出格式：

```json
{{
  "answer": 答案
}}
```
        """
        )

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=2, max=30),
        stop=tenacity.stop_after_attempt(5),
        retry=tenacity.retry_if_exception(is_retryable_error),
        before_sleep=tenacity.before_sleep_log(logger, logging.INFO),
        reraise=True,
    )
    def query(self, user_input: str):
        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                config={"response_mime_type": "application/json"},
                contents=self.system_prompt.format(user_input=user_input),
            )

            result = robust_json_parse(response.text)
            return {"answer": result.get("answer", "No answer found in response.")}
        except Exception as e:
            if is_retryable_error(e):
                raise e
            logger.error(f"Non-retryable error in query: {e}")
            return {"answer": f"Error: {str(e)}"}

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=2, max=30),
        stop=tenacity.stop_after_attempt(5),
        retry=tenacity.retry_if_exception(is_retryable_error),
        reraise=True,
    )
    async def aquery(self, user_input: str):
        try:
            response = await self.client.aio.models.generate_content(
                model="gemini-2.0-flash",
                config={"response_mime_type": "application/json"},
                contents=self.system_prompt.format(user_input=user_input),
            )

            result = robust_json_parse(response.text)
            return {"answer": result.get("answer", "No answer found in response.")}
        except Exception as e:
            if is_retryable_error(e):
                raise e
            logger.error(f"Non-retryable error in aquery: {e}")
            return {"answer": f"Error during aquery: {str(e)}"}


def default_rag_client(
    llm_client, model_name: str = "gemini-2.5-pro", documents: Optional[List[str]] = None, tokenizer: str = "split"
) -> SimpleRAG:
    """
    Create a default RAG client with OpenAI LLM and optional retriever.

    Args:
        llm_client: LLM client with a generate() method
        model_name: Default model to use for generation
        documents: Optional list of documents to add
        tokenizer: Tokenizer to use for keyword matching ("split" or "jieba")
    Returns:
        ExampleRAG instance
    """
    retriever = SimpleKeywordRetriever(tokenizer=tokenizer)
    client = SimpleRAG(llm_client=llm_client, retriever=retriever, model_name=model_name)
    if documents:
        client.add_documents(documents)
    return client


