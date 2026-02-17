# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: rag4pdf
#     language: python
#     name: rag4pdf
# ---

# %% [markdown]
# # Simple Rag

# %%
# %reload_ext viztracer

# %%
import os
from pathlib import Path

# 替换为你的代理端口，通常是 7890, 7897, 1080 等
proxy = "http://127.0.0.1:7890" 

if proxy:
    os.environ["http_proxy"] = proxy
    os.environ["https_proxy"] = proxy
    os.environ["HTTP_PROXY"] = proxy
    os.environ["HTTPS_PROXY"] = proxy

# %%
# 加载 .env 文件中的环境变量
from dotenv import load_dotenv
load_dotenv()

import logging
logging.basicConfig(level=logging.WARNING)

# %%
under_n_chars = 1000
out_dir = "../outputs"
book = "数据仓库工具箱维度建模权威指南（第3版）"
chapter = "2" # 可以设为 "*" 来匹配所有章节
book_fmt = "md"

llm_model = "gemini-2.0-flash"
embedding_model = "models/gemini-embedding-001"

testset_size = 10
query_dist = None

# 使用 Glob 模式匹配文件
in_path_pattern = f"{out_dir}/{book}/{chapter}.{book_fmt}"
kg_path = f"{out_dir}/{book}/{chapter}-kg-cn-{under_n_chars}.json"
eval_path = f"{out_dir}/{book}/{chapter}-eval-cn-{under_n_chars}.json"

# %%
# 使用 Glob 加载多个 Markdown 文件并转换为 Langchain Document
from langchain_core.documents import Document
import glob

documents = []
files = glob.glob(in_path_pattern)

for file_path in files:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    documents.append(Document(page_content=content, metadata={"source": file_path}))

print(f"成功从 {in_path_pattern} 匹配到 {len(files)} 个文件，共转换为 {len(documents)} 个 Document 对象")

# %%
# 文档切分 (Chunking)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 配置切分参数
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=under_n_chars,       # 每个 chunk 的字符数
    chunk_overlap=100,     # chunk 之间的重叠字符数
    length_function=len,
    is_separator_regex=False,
)

# 执行切分
chunked_documents = text_splitter.split_documents(documents)

print(f"切分完成：从 {len(documents)} 个原始文档生成了 {len(chunked_documents)} 个 Chunks")

# 将后续使用的 documents 变量更新为切分后的版本
# 注意：这会影响后续 KnowledgeGraph 的构建
documents = chunked_documents

# %% [markdown]
# ## 2. 评测集准备
# ### 2.1. 初始化 LLM

# %%
"""
from ragas.cache import DiskCacheBackend

# 1. 开启本地磁盘缓存
# 这会在当前目录下创建一个 .ragas_cache 文件夹
cache = DiskCacheBackend()

# 2. 初始化大模型客户端
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
gemini_llm = ChatGoogleGenerativeAI(model=llm_model)
gemini_embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)

# 3. 构建工厂类，并使用缓存
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
generator_llm = LangchainLLMWrapper(gemini_llm, cache=cache)
embeddings = LangchainEmbeddingsWrapper(gemini_embeddings, cache=cache)
"""

# %%
from ragas.cache import DiskCacheBackend
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from google import genai
from rag4pdf import GoogleGenAIWrapper

# 1. 开启本地磁盘缓存
cache = DiskCacheBackend()

# 2. 初始化大模型客户端
client = genai.Client()

# 3. 构建工厂类，并使用缓存
generator_llm = GoogleGenAIWrapper(client=client, model=llm_model, cache=cache)
embeddings = embedding_factory(provider="google", model=embedding_model, client=client, cache=cache)

# %% [markdown]
# ### 2.2. 生成评测集

# %%
from ragas.run_config import RunConfig

# 定义一个适配低配额 API 的配置
run_config = RunConfig(
    max_workers=5,      # 核心设置：降低并发
    max_retries=10,     # 遇到 429 时自动重试
    timeout=180         # 给 Gemini 留出足够的生成时间
)

# %%
"""
# v0.2 版本
from ragas.run_config import RunConfig
from ragas.testset import TestsetGenerator

# 1. 构建 Ragas 生成器
generator = TestsetGenerator.from_langchain(
    generator_llm,
    embeddings
)

# 针对 Unstructured 的文档特征生成数据集
# 创建一个低并发的运行配置
# max_workers=1 表示串行，最稳妥但慢一点
# max_retries=10 增加重试次数，应对短暂的 429
run_config = RunConfig(
    max_workers=1, 
    max_retries=10, 
    timeout=60
)

testset = generator.generate_with_chunks(
    documents, 
    testset_size=10,
    query_distribution={
        "simple": 0.4, 
        "reasoning": 0.4, 
        "multi_context": 0.2
    },
    run_config=run_config
)

# 导出结果
df = testset.to_pandas()
"""

# %% [markdown]
# #### 2.2.1. 知识图谱生成（为提升评测集质量）

# %%
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import default_transforms, apply_transforms

# 1. 将 Langchain Documents 转换为 Ragas Nodes，并初始化知识图谱
nodes = [Node(
    type=NodeType.DOCUMENT,
    properties={"page_content": doc.page_content}
) for doc in documents]

kg = KnowledgeGraph(
    nodes=nodes
)

# 2. 配置并运行提取器 (这里会产生 LLM 消耗)
# 如果你的 25% 文档 token 超过一百，默认会把文档再进行 Title 拆封
# 需要考虑缓存
transformers = default_transforms(documents, generator_llm, embeddings)

# 开始构建
apply_transforms(kg, transformers, run_config=run_config)

# 4. 【关键步骤】保存图谱到本地
kg.save(kg_path)

# %% [markdown]
# #### 2.2.2. 生成评测集

# %%
from ragas.testset import TestsetGenerator

# 2. 初始化生成器 (此时它已拥有所有向量和实体信息)
from ragas.testset.graph import KnowledgeGraph
kg = KnowledgeGraph.load(kg_path)
generator = TestsetGenerator(generator_llm, embeddings, knowledge_graph=kg)

# 3. 生成题目 (因为不用再做 NER 和 Embedding，这步很快且报错率极低)

# 如果你希望简单省事，可以直接使用默认分布函数
from ragas.testset.synthesizers import default_query_distribution
from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer
query_distribution = [
    (SingleHopSpecificQuerySynthesizer(llm=generator_llm), 1.0)
] if query_dist is not None else default_query_distribution(generator_llm)

testset = generator.generate(
    testset_size=testset_size,
    query_distribution=query_distribution,
    run_config=run_config
    # with_debugging_logs=True
)

df = testset.to_pandas()

# %%
df

# %%
eval_dataset = testset.to_evaluation_dataset()
n = 7
print("Query:", eval_dataset[n].user_input)
print("Context:")
for c in eval_dataset[n].reference_contexts:
    print(c)
print("Reference:", eval_dataset[n].reference)

# %%
df.to_json(eval_path)

# %% [markdown]
# ## 3. 评测实验

# %% [markdown]
# ### 3.1 载入评测集

# %%
from ragas import Dataset

# Create a new dataset
dataset = Dataset(name="维度建模", backend="local/jsonl", root_dir="./ragas_data")

# %%
import pandas as pd

df = pd.read_json(eval_path)

for _, row in df.iterrows():
    dataset.append({"user_input": row["user_input"], "reference": row["reference"]})

dataset.save()

# %%
dataset.to_pandas()

# %% [markdown]
# ### 3.2 定义 RAG

# %%
from google import genai
from rag4pdf import default_rag_client
api_key = os.environ["GOOGLE_API_KEY"]
client = genai.Client(api_key=api_key)

# %%
import asyncio
import json
import logging
import tenacity
import re
from google.genai import errors
from rag4pdf import SimpleRAG, robust_json_parse, is_retryable_error

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化 SimpleRAG，指定使用 gemini-2.0-flash
rag_client = default_rag_client(llm_client=client, model_name="gemini-2.0-flash", tokenizer="jieba")

# 将笔记本中之前切分好的文档内容加载到 RAG 中
# 注意：documents 变量中存储的是 Langchain 的 Document 对象，需要提取 page_content
rag_client.add_documents([doc.page_content for doc in documents])

print(f"SimpleRAG 初始化完成，已加载 {len(documents)} 个文档切片")


# %%
dataset = Dataset.load(name="维度建模", backend="local/jsonl", root_dir="../ragas_data")
d = dataset[3]
user_input = d["user_input"]
pred = rag_client.query(user_input)
print(pred)

# %% [markdown]
# ### 3.3 构建指标(Metric)

# %%
ANSWER_CORRECTNESS_PROMPT_SIMPLE = """评估生成答案相对于参考答案的准确性。

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


# %%
from ragas.metrics import numeric_metric
from ragas.metrics.result import MetricResult
import tenacity
import json
import logging
from google.genai import errors
from rag4pdf.utils import is_retryable_error, robust_json_parse
from rag4pdf.eval import correctness_metric, faithfulness_metric, answer_relevance_metric

logger = logging.getLogger(__name__)

# 指标已经在 rag4pdf.eval 中定义，直接使用即可。
# 如果需要自定义指标，可以参考 eval.py 中的实现。

# %%
dataset = Dataset.load(name="维度建模", backend="local/jsonl", root_dir="../ragas_data")
d = dataset[3]
user_input = d["user_input"]
reference = d["reference"]
prediction = "根据提供的文档，“确认维度”是 **4步骤维度设计过程** 的一部分。 这个过程包含4个主要决策： 1. 选择业务过程 2. 声明粒度 3. **确认维度** 4. 确认事实"
print(ANSWER_CORRECTNESS_PROMPT_SIMPLE.format(user_input=user_input, reference=reference, prediction=prediction))

# %%
import time
start = time.time()
# Passing the global client
await correctness_metric.ascore(user_input=user_input, reference=reference, prediction=prediction, client=client)
print(f"Correctness metric took: {time.time() - start:.2f}s")

# %% [markdown]
# ### 3.4 构建实验(Experiment)

# %%
import time
from ragas import experiment

@experiment()
async def run_experiment(row):
    user_input = row["user_input"]
    reference = row["reference"]
    
    response = await rag_client.aquery(user_input)
    prediction = response["answer"]
    retrieved_docs = response.get("retrieved_docs", [])
    
    # 提取召回结果作为 context
    context = "\n\n".join([doc["content"] for doc in retrieved_docs])

    # Calculate metrics
    # Passing the global client to metrics
    correctness = await correctness_metric.ascore(user_input=user_input, reference=reference, prediction=prediction, client=client)
    faithfulness = await faithfulness_metric.ascore(context=context, prediction=prediction, client=client)
    answer_relevance = await answer_relevance_metric.ascore(user_input=user_input, prediction=prediction, client=client)

    return {
        "expression": user_input,
        "expected_result": reference,
        "result": prediction,
        "retrieved_docs": retrieved_docs,
        "correctness": correctness.value,
        "correctness_reason": correctness.reason,
        "faithfulness": faithfulness.value,
        "faithfulness_reason": faithfulness.reason,
        "answer_relevance": answer_relevance.value,
        "answer_relevance_reason": answer_relevance.reason
    }


# %% [markdown]
# ### 3.5 开始评测(Evaluation)

# %%
async def run_evaluation():
    dataset = Dataset.load(name="维度建模", backend="local/jsonl", root_dir="../ragas_data")
    #logging.info("Experiment started successfully!")
    #start = time.time()
    experiment_results: Experiment = await run_experiment.arun(dataset)
    #logging.info(f"Experiment completed successfully! Total: {time.time() - start:.2f}s")
    # Save experiment results to CSV
    experiment_results.save()
    jsonl_path: Path = Path("./ragas_data") / "experiments" / f"{experiment_results.name}.jsonl"
    logging.info(f"\nExperiment results saved to: {jsonl_path.resolve()}")

    # Print results
    # if experiment_results:
    #    score = sum(result.get("correctness") or 0 for result in experiment_results)
    #    total_count = len(experiment_results)
    #    correctness = round(score / total_count) if total_count > 0 else 0
    #    print(f"Results: {correctness}/{total_count}")

    return experiment_results


# %%
# # %%viztracer
import asyncio
experiment_results = await run_evaluation()

# %%
experiment_results.to_pandas()

# %%
# Print results
print(experiment_results)
if experiment_results:
    total_count = len(experiment_results)
    
    correctness_score = sum(result.get("correctness") or 0 for result in experiment_results)
    correctness = round(correctness_score / total_count, 2) if total_count > 0 else 0
    
    relevance_score = sum(result.get("answer_relevance") or 0 for result in experiment_results)
    relevance = round(relevance_score / total_count, 2) if total_count > 0 else 0
    
    faithfulness_score = sum(result.get("faithfulness") or 0 for result in experiment_results)
    faithfulness = round(faithfulness_score / total_count, 2) if total_count > 0 else 0

    print(f"\nEvaluation Results (Average Score):")
    print(f"Correctness: {correctness}")
    print(f"Answer Relevance: {relevance}")
    print(f"Faithfulness: {faithfulness}")
    print(f"Total samples: {total_count}")


# %%
