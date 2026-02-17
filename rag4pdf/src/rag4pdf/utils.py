import json
import re
import logging
from google.genai import errors

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
