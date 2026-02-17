import pytest
from unittest.mock import MagicMock
from rag4pdf.service import SimpleRAG, SimpleKeywordRetriever, SimpleAgent

DOCUMENTS = [
    "Ragas are melodic frameworks in Indian classical music.",
    "There are many types of ragas, each with its own mood and time of day.",
    "Ragas are used to evoke specific emotions in the listener.",
    "The performance of a raga involves improvisation within a set structure.",
    "Ragas can be performed on various instruments or sung vocally.",
]

def test_rag_query_flow():
    # Mock the LLM client
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Ragas are melodic frameworks used in Indian classical music."
    mock_llm.models.generate_content.return_value = mock_response

    # Initialize RAG system
    retriever = SimpleKeywordRetriever()
    rag_client = SimpleRAG(llm_client=mock_llm, retriever=retriever)

    # Add documents
    rag_client.add_documents(DOCUMENTS)

    # Run query
    query = "What is Ragas"
    response = rag_client.query(query, top_k=3)

    # Verify the response structure and content
    assert "answer" in response
    assert "run_id" in response
    assert response["answer"] == "Ragas are melodic frameworks used in Indian classical music."

    # Verify that the LLM was called
    mock_llm.models.generate_content.assert_called_once()

def test_simple_agent_query():
    # Mock the LLM client
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '```json\n{"answer": "Paris is the capital of France."}\n```'
    mock_llm.models.generate_content.return_value = mock_response

    agent = SimpleAgent(client=mock_llm)
    result = agent.query("What is the capital of France?")

    assert result["answer"] == "Paris is the capital of France."
    mock_llm.models.generate_content.assert_called_once()

def test_retriever_ranking():
    retriever = SimpleKeywordRetriever()
    retriever.fit(DOCUMENTS)

    query = "melodic frameworks"
    top_docs = retriever.get_top_k(query, k=1)

    assert top_docs[0][0] == 0
    assert top_docs[0][1] >= 2
