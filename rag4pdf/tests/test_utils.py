import json
import pytest
from rag4pdf.utils import is_retryable_error, robust_json_parse
from google.genai import errors

def test_is_retryable_error():
    # 429 and 503 errors
    assert is_retryable_error(Exception("Error 429: Rate limit exceeded")) is True
    assert is_retryable_error(Exception("503 Service Unavailable")) is True
    assert is_retryable_error(Exception("quota exceeded")) is True

    # SDK errors
    assert is_retryable_error(errors.ClientError("Client side error", response_json={})) is True
    assert is_retryable_error(errors.ServerError("Server side error", response_json={})) is True

    # JSON decode errors
    assert is_retryable_error(json.JSONDecodeError("msg", "doc", 0)) is True

    # Non-retryable errors
    assert is_retryable_error(Exception("Something went wrong")) is False
    assert is_retryable_error(ValueError("Invalid value")) is True # ValueError is marked as retryable in our implementation

def test_robust_json_parse():
    # Direct JSON
    data = '{"key": "value"}'
    assert robust_json_parse(data) == {"key": "value"}

    # Markdown JSON block
    data_md = 'Some text before\n```json\n{"key": "value"}\n```\nSome text after'
    assert robust_json_parse(data_md) == {"key": "value"}

    # JSON with markdown block but no language tag
    data_md_no_tag = '```\n{"key": "value"}\n```'
    assert robust_json_parse(data_md_no_tag) == {"key": "value"}

    # JSON embedded in text
    data_embedded = 'The answer is {"key": "value"} hope that helps.'
    assert robust_json_parse(data_embedded) == {"key": "value"}

    # Empty string
    with pytest.raises(ValueError, match="Empty response"):
        robust_json_parse("")

    # Invalid JSON
    with pytest.raises(ValueError, match="Failed to parse JSON"):
        robust_json_parse("This is not JSON")
