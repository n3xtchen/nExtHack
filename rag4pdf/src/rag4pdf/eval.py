import logging
import typing as t
from pathlib import Path

import pandas as pd
import tenacity
from google.genai import errors
from langchain_core.outputs import Generation, LLMResult
from ragas import Dataset, experiment
from ragas.cache import CacheInterface
from ragas.llms import BaseRagasLLM
from ragas.metrics import numeric_metric
from ragas.metrics.result import MetricResult
from ragas.run_config import RunConfig
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph

from .utils import is_retryable_error, robust_json_parse

logger = logging.getLogger(__name__)

class GoogleGenAIWrapper(BaseRagasLLM):
    """
    Ragas 专用的 Google GenAI LLM 适配器。
    """
    def __init__(self,
        client,
        model: str = "gemini-2.0-flash",
        run_config: t.Optional[RunConfig] = None,
        cache: t.Optional[CacheInterface] = None,
    ):
        super().__init__(cache=cache)
        self.client = client
        self.model = model
        self.run_config = run_config or RunConfig()

    def generate_text(self, prompt: t.Any, **kwargs) -> LLMResult:
        prompt_str = str(prompt)
        if "chinese" not in prompt_str.lower():
             prompt_str += "\n\n请使用中文回答 (Please respond in Chinese)."

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt_str
        )
        return LLMResult(generations=[[Generation(text=response.text)]])

    async def agenerate_text(self, prompt: t.Any, **kwargs) -> LLMResult:
        prompt_str = str(prompt)
        if "chinese" not in prompt_str.lower():
             prompt_str += "\n\n请使用中文回答 (Please respond in Chinese)."

        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=prompt_str
        )
        return LLMResult(generations=[[Generation(text=response.text)]])

    def is_finished(self, response):
        return True


def get_testset_generator(llm, embeddings, kg_path: str) -> TestsetGenerator:
    """
    获取测试集生成器。
    """
    kg = KnowledgeGraph.load(kg_path)
    return TestsetGenerator(llm, embeddings, knowledge_graph=kg)


# --- 评测指标定义 ---

ANSWER_CORRECTNESS_PROMPT = """评估生成答案相对于参考答案的准确性。

**问题**：{user_input}

**参考答案**：{reference}

**生成答案**：{prediction}

**评估要求**：
1. 检查事实是否正确
2. 检查是否包含所有关键信息
3. 检查语义是否一致

给出 0-5 分数（5=完美，0=完全错误）和简短理由。

请使用中文回答 (Please respond in Chinese).

输出格式：
```json
{{
  "score": <分数>,
  "reasoning": "<理由>"
}}
```
"""

FAITHFULNESS_PROMPT = """评估生成的答案是否完全由提供的上下文（检索到的文档）支持。

**上下文**：
{context}

**生成的答案**：
{prediction}

**评估要求**：
1. 检查答案中的每个陈述是否都能在上下文中找到依据。
2. 如果答案包含上下文之外的信息（即使在现实中是正确的），则在“支持度”评分中应予以扣分，因为这可能暗示了模型幻觉或未对齐。
3. 给出 0-5 分数（5=完全由上下文支持，0=与上下文无关或完全是幻觉）和理由。

请使用中文回答 (Please respond in Chinese).

输出格式：
```json
{{
  "score": <分数>,
  "reasoning": "<理由>"
}}
```
"""

ANSWER_RELEVANCE_PROMPT = """评估生成的答案与原始问题的相关性。

**问题**：{user_input}

**生成的答案**：{prediction}

**评估要求**：
1. 评估答案是否直接回答了问题。
2. 检查答案是否包含冗余、无关或答非所问的信息。
3. 即使答案在事实上是正确的，如果它没有直接针对问题，也应该给低分。
4. 给出 0-5 分数（5=高度相关且直接，0=完全不相关或答非所问）和理由。

请使用中文回答 (Please respond in Chinese).

输出格式：
```json
{{
  "score": <分数>,
  "reasoning": "<理由>"
}}
```
"""

@numeric_metric(name="correctness", allowed_values=(0.0, 5.0))
async def correctness_metric(user_input: str, reference: str, prediction: str, client, model_name="gemini-2.0-flash"):
    """使用 LLM 作为裁判进行评分。"""
    if isinstance(prediction, str) and ("ERROR" in prediction or "Error" in prediction):
        return MetricResult(value=0.0, reason=f"预测出错: {prediction}")

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=2, max=30),
        stop=tenacity.stop_after_attempt(5),
        retry=tenacity.retry_if_exception(is_retryable_error),
        reraise=True
    )
    async def _acall_llm():
        return await client.aio.models.generate_content(
            model=model_name,
            config={"response_mime_type": "application/json"},
            contents=ANSWER_CORRECTNESS_PROMPT.format(user_input=user_input, reference=reference, prediction=prediction)
        )

    try:
        response = await _acall_llm()
        judge_result = robust_json_parse(response.text)
        return MetricResult(value=float(judge_result["score"]), reason=judge_result["reasoning"])
    except Exception as e:
        logger.warning(f"Correctness metric failed: {e}")
        return MetricResult(value=0.0, reason=str(e))


@numeric_metric(name="faithfulness", allowed_values=(0.0, 5.0))
async def faithfulness_metric(context: str, prediction: str, client, model_name="gemini-2.0-flash"):
    """评估答案是否忠实于上下文。"""
    if not context:
        return MetricResult(value=0.0, reason="无上下文")

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=2, max=30),
        stop=tenacity.stop_after_attempt(5),
        retry=tenacity.retry_if_exception(is_retryable_error),
        reraise=True
    )
    async def _acall_llm():
        return await client.aio.models.generate_content(
            model=model_name,
            config={"response_mime_type": "application/json"},
            contents=FAITHFULNESS_PROMPT.format(context=context, prediction=prediction)
        )

    try:
        response = await _acall_llm()
        judge_result = robust_json_parse(response.text)
        return MetricResult(value=float(judge_result["score"]), reason=judge_result["reasoning"])
    except Exception as e:
        logger.warning(f"Faithfulness metric failed: {e}")
        return MetricResult(value=0.0, reason=str(e))


@numeric_metric(name="answer_relevance", allowed_values=(0.0, 5.0))
async def answer_relevance_metric(user_input: str, prediction: str, client, model_name="gemini-2.0-flash"):
    """评估答案与问题的相关性。"""
    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=2, max=30),
        stop=tenacity.stop_after_attempt(5),
        retry=tenacity.retry_if_exception(is_retryable_error),
        reraise=True
    )
    async def _acall_llm():
        return await client.aio.models.generate_content(
            model=model_name,
            config={"response_mime_type": "application/json"},
            contents=ANSWER_RELEVANCE_PROMPT.format(user_input=user_input, prediction=prediction)
        )

    try:
        response = await _acall_llm()
        judge_result = robust_json_parse(response.text)
        return MetricResult(value=float(judge_result["score"]), reason=judge_result["reasoning"])
    except Exception as e:
        logger.warning(f"Answer relevance metric failed: {e}")
        return MetricResult(value=0.0, reason=str(e))


@experiment()
async def run_experiment(row, rag_client, eval_client, eval_model="gemini-2.0-flash"):
    """
    运行单个样本的评测实验。
    """
    user_input = row["user_input"]
    reference = row["reference"]

    response = await rag_client.aquery(user_input)
    prediction = response["answer"]
    retrieved_docs = response.get("retrieved_docs", [])
    context = "\n\n".join([doc["content"] for doc in retrieved_docs])

    # Calculate metrics
    c_res = await correctness_metric.ascore(user_input=user_input, reference=reference, prediction=prediction, client=eval_client, model_name=eval_model)
    f_res = await faithfulness_metric.ascore(context=context, prediction=prediction, client=eval_client, model_name=eval_model)
    r_res = await answer_relevance_metric.ascore(user_input=user_input, prediction=prediction, client=eval_client, model_name=eval_model)

    return {
        "user_input": user_input,
        "reference": reference,
        "prediction": prediction,
        "correctness": c_res.value,
        "correctness_reason": c_res.reason,
        "faithfulness": f_res.value,
        "faithfulness_reason": f_res.reason,
        "answer_relevance": r_res.value,
        "answer_relevance_reason": r_res.reason
    }


async def run_evaluation(dataset_name: str, rag_client, eval_client, root_dir="./ragas_data", eval_model="gemini-2.0-flash"):
    """
    运行完整数据集的评测并保存结果。
    """
    dataset = Dataset.load(name=dataset_name, backend="local/jsonl", root_dir=root_dir)
    experiment_results = await run_experiment.arun(dataset, rag_client=rag_client, eval_client=eval_client, eval_model=eval_model)
    experiment_results.save()

    jsonl_path = Path(root_dir) / "experiments" / f"{experiment_results.name}.jsonl"
    logger.info(f"Evaluation results saved to: {jsonl_path}")

    return experiment_results
