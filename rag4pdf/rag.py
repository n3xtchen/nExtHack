import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from google import genai

DOCUMENTS = [
    "Ragas are melodic frameworks in Indian classical music.",
    "There are many types of ragas, each with its own mood and time of day.",
    "Ragas are used to evoke specific emotions in the listener.",
    "The performance of a raga involves improvisation within a set structure.",
    "Ragas can be performed on various instruments or sung vocally.",
]


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

    def __init__(self):
        super().__init__()

    def _count_keyword_matches(self, query: str, document: str) -> int:
        """Count how many query words appear in the document"""
        query_words = query.lower().split()
        document_words = document.lower().split()
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

    def generate_response(self, query: str, top_k: int = 3, model_name: Optional[str] = None) -> str:
        """
        Generate response to query using retrieved documents

        Args:
            query: User query
            top_k: Number of documents to retrieve
            model_name: Optional model to use for this specific request

        Returns:
            Generated response
        """
        if not self.is_fitted:
            raise ValueError(
                "No documents have been added. Call add_documents() or set_documents() first."
            )

        # Retrieve relevant documents
        retrieved_docs = self.retrieve_documents(query, top_k)

        if not retrieved_docs:
            return "I couldn't find any relevant documents to answer your question."

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
            return response.text.strip()

        except Exception as e:
            return f"Error generating response: {str(e)}"

    def query(
        self, question: str, top_k: int = 3, run_id: Optional[str] = None, model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve documents and generate response

        Args:
            question: User question
            top_k: Number of documents to retrieve
            run_id: Optional run ID for tracing (auto-generated if not provided)
            model_name: Optional model to use for this specific request

        Returns:
            Dictionary containing response and run_id
        """
        # Generate run_id if not provided
        if run_id is None:
            run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(question) % 10000:04d}"

        try:
            response = self.generate_response(question, top_k, model_name=model_name)
            return {"answer": response, "run_id": run_id}

        except Exception as e:
            # Return error result
            return {
                "answer": f"Error processing query: {str(e)}",
                "run_id": run_id,
            }


def default_rag_client(llm_client, model_name: str = "gemini-2.5-pro") -> SimpleRAG:
    """
    Create a default RAG client with OpenAI LLM and optional retriever.

    Args:
        llm_client: LLM client with a generate() method
        model_name: Default model to use for generation
    Returns:
        ExampleRAG instance
    """
    retriever = SimpleKeywordRetriever()
    client = SimpleRAG(llm_client=llm_client, retriever=retriever, model_name=model_name)
    client.add_documents(DOCUMENTS)  # Add default documents
    return client


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    try:
        api_key = os.environ["GOOGLE_API_KEY"]
    except KeyError:
        print("Error: GOOGLE_API_KEY environment variable is not set.")
        print("Please set your Google API key in .env file:")
        print("GOOGLE_API_KEY='your_google_api_key'")
        exit(1)

    # Initialize RAG system
    client = genai.Client(api_key=api_key)
    r = SimpleKeywordRetriever()
    rag_client = SimpleRAG(llm_client=client, retriever=r)

    # Add documents
    rag_client.add_documents(DOCUMENTS)

    # Run query
    query = "What is Ragas"
    print(f"Query: {query}")
    response = rag_client.query(query, top_k=3)

    print("Response:", response["answer"])
    print(f"Run ID: {response['run_id']}")
