import pytest
from unittest.mock import MagicMock, AsyncMock
from rag4pdf.eval import GoogleGenAIWrapper, correctness_metric, faithfulness_metric, answer_relevance_metric

class MockResponse:
    def __init__(self, text):
        self.text = text

class MockModels:
    def __init__(self, response_text):
        self.generate_content = MagicMock(return_value=MockResponse(response_text))
        self.generate_content_async = AsyncMock(return_value=MockResponse(response_text))

class MockAio:
    def __init__(self, response_text):
        self.models = MagicMock()
        self.models.generate_content = AsyncMock(return_value=MockResponse(response_text))

class MockClient:
    def __init__(self, response_text):
        self.models = MockModels(response_text)
        self.aio = MockAio(response_text)

@pytest.mark.asyncio
async def test_google_genai_wrapper():
    client = MockClient("Hello from Gemini")
    wrapper = GoogleGenAIWrapper(client=client)

    # Test sync
    res_sync = wrapper.generate_text("Hi")
    assert res_sync.generations[0][0].text == "Hello from Gemini"

    # Test async
    res_async = await wrapper.agenerate_text("Hi")
    assert res_async.generations[0][0].text == "Hello from Gemini"

@pytest.mark.asyncio
async def test_correctness_metric():
    response_text = '```json\n{"score": 4.5, "reasoning": "Very accurate"}\n```'
    client = MockClient(response_text)

    result = await correctness_metric.ascore(
        user_input="What is RAG?",
        reference="Retrieval Augmented Generation",
        prediction="RAG stands for Retrieval Augmented Generation",
        client=client
    )

    assert result.value == 4.5
    assert result.reason == "Very accurate"

@pytest.mark.asyncio
async def test_faithfulness_metric():
    response_text = '{"score": 5.0, "reasoning": "Fully supported"}'
    client = MockClient(response_text)

    result = await faithfulness_metric.ascore(
        context="RAG uses a retriever and a generator.",
        prediction="RAG involves retrieval and generation.",
        client=client
    )

    assert result.value == 5.0
    assert result.reason == "Fully supported"

@pytest.mark.asyncio
async def test_answer_relevance_metric():
    response_text = '{"score": 4.0, "reasoning": "Highly relevant but slightly long"}'
    client = MockClient(response_text)

    result = await answer_relevance_metric.ascore(
        user_input="What is RAG?",
        prediction="Retrieval-Augmented Generation (RAG) is a technique for enhancing LLMs with external knowledge.",
        client=client
    )

    assert result.value == 4.0
    assert result.reason == "Highly relevant but slightly long"
