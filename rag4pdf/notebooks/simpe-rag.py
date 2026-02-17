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
from rag4pdf import (
    load_markdown_documents,
    split_documents,
    build_knowledge_graph,
    default_rag_client,
    GoogleGenAIWrapper,
    # get_testset_generator
)
from rag4pdf.eval import (
    correctness_metric,
    faithfulness_metric,
    answer_relevance_metric,
    run_evaluation
)

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
# 使用库函数加载 Markdown 文件
documents = load_markdown_documents(in_path_pattern)
print(f"成功从 {in_path_pattern} 加载了 {len(documents)} 个 Document 对象")

# %%
# 使用库函数进行文档切分 (Chunking)
chunked_documents = split_documents(documents, chunk_size=under_n_chars)
print(f"切分完成：从 {len(documents)} 个原始文档生成了 {len(chunked_documents)} 个 Chunks")

# 将后续使用的 documents 变量更新为切分后的版本
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
from google import genai

# 1. 开启本地磁盘缓存
cache = DiskCacheBackend()

# 2. 初始化大模型客户端
client = genai.Client()

# 3. 构建 Ragas 适配器，并使用缓存
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
# 使用库函数构建并保存知识图谱
kg = build_knowledge_graph(
    documents,
    generator_llm,
    embeddings,
    kg_path,
    run_config=run_config
)

# %%
# 保存图谱到本地
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
# 使用库函数初始化 SimpleRAG
rag_client = default_rag_client(llm_client=client, model_name=llm_model, tokenizer="jieba")

# 将文档内容加载到 RAG 中
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
import time
start = time.time()
# Passing the global client
await correctness_metric.ascore(user_input=user_input, reference=reference, prediction=prediction, client=client)
print(f"Correctness metric took: {time.time() - start:.2f}s")

# %% [markdown]
# ### 3.4 构建实验(Experiment)

# %%
# run_experiment 已从 rag4pdf.eval 导入
pass

# %% [markdown]
# ### 3.5 开始评测(Evaluation)

# %%
# run_evaluation 已从 rag4pdf.eval 导入
pass

# %%
# # %%viztracer
import asyncio
# 使用库函数运行完整评测
experiment_results = await run_evaluation(
    dataset_name="维度建模",
    rag_client=rag_client,
    eval_client=client,
    root_dir="../ragas_data",
    eval_model=llm_model
)

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
