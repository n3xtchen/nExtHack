import pytest
from unittest.mock import MagicMock
from rag import SimpleRAG, SimpleKeywordRetriever

DOCUMENTS = [
    "Ragas are melodic frameworks in Indian classical music.",
    "There are many types of ragas, each with its own mood and time of day.",
    "Ragas are used to evoke specific emotions in the listener.",
    "The performance of a raga involves improvisation within a set structure.",
    "Ragas can be performed on various instruments or sung vocally.",
]

def test_rag_query_flow():
    """
    Test the full flow of SimpleRAG:
    1. Initialize with mocked LLM
    2. Add documents
    3. Query
    4. Verify response and interactions
    """
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

    # Verify that the LLM was called with expected prompt containing context
    mock_llm.models.generate_content.assert_called_once()

    # Inspect arguments passed to the LLM
    call_args = mock_llm.models.generate_content.call_args
    kwargs = call_args.kwargs
    prompt_content = kwargs.get('contents', '')

    # Check if prompt contains the query
    assert f"问题: {query}" in prompt_content

    # Check if prompt contains retrieved context
    assert "Ragas are melodic frameworks in Indian classical music." in prompt_content

def test_retriever_ranking():
    """Test that the retriever correctly ranks documents based on keyword matches"""
    retriever = SimpleKeywordRetriever()
    retriever.fit(DOCUMENTS)

    query = "melodic frameworks"
    top_docs = retriever.get_top_k(query, k=1)

    # The first document contains "melodic frameworks"
    # index 0: "Ragas are melodic frameworks in Indian classical music."
    assert top_docs[0][0] == 0
    assert top_docs[0][1] >= 2  # At least 2 matches
