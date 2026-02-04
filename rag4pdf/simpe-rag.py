# ---
# jupyter:
#   jupytext:
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
import os

# 替换为你的代理端口，通常是 7890, 7897, 1080 等
proxy = "http://127.0.0.1:7890" 

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
under_n_chars = 100
out_dir = "outputs"
book = "数据仓库工具箱维度建模权威指南（第3版）"
chapter = "2"
book_fmt = "md"

llm_model = "gemini-2.5-pro"
embedding_model = "models/embedding-001"

testset_size = 10
query_dist = None

in_path = f"{out_dir}/{book}/{chapter}.{book_fmt}"
kg_path = f"{out_dir}/{book}/{chapter}-kg-{under_n_chars}.json"
eval_path = f"{out_dir}/{book}/{chapter}-eval-{under_n_chars}.json"

# %% [markdown]
# ## 1. 分块

# %%
from unstructured.partition.md import partition_md
from unstructured.chunking.title import chunk_by_title

# 1. 将 Markdown 文件解析为 Element 对象列表
# partition_md 会自动识别标题（Title）、正文（NarrativeText）、列表（ListItem）等
elements = partition_md(filename=in_path)

# 2. 按照标题进行分块
# chunk_by_title 会在遇到新的标题（Title 元素）时启动一个新块
chunks = chunk_by_title(
    elements,
    combine_text_under_n_chars=under_n_chars,  # 是否合并较小的块，设为0表示严格按标题切分
    max_characters=1000,           # 每个分块的最大字符数（硬限制）
    new_after_n_chars=500,         # 超过此字数后，遇到新的标题立即切分（软限制）
    multipage_sections=True        # 是否允许块跨越页面（对 Markdown 而言通常设为 True）
)

# 3. 分块处理
current_path = {1: "", 2: "", 3: ""}
documents = []

class DocumentWrapper:
    def __init__(self, content, metadata):
        self.page_content = content
        self.metadata = metadata
        
for i, chunk in enumerate(chunks):
    for el in chunk.metadata.orig_elements:
        if el.category == "Title":
            # 获取当前标题层级 (category_depth: 0=H1, 1=H2...)
            depth = getattr(el.metadata, "category_depth", 0) + 1
            current_path[depth] = el.text
            # 当 H2 更新时，清空 H3, H4
            for i in range(depth + 1, 3):
                current_path[i] = ""

    # 构建完整路径字符串: "章 > 节 > 小节"
    if len(current_path) > 0:
        breadcrumb = " ".join([v for v in current_path.values() if v][:-1])
    else:
        breadcrumb = ''
        
    documents.append(DocumentWrapper(f"{breadcrumb} {" ".join(chunk.text.split())}\n", {
        "filename": in_path, "page_num": i
    }))

print(f"Chunk Num: {i}")

# %% [markdown]
# ## 2. 评测集准备
# ### 2.1. 初始化 LLM

# %%
from ragas.cache import DiskCacheBackend
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory

# 1. 开启本地磁盘缓存
# 这会在当前目录下创建一个 .ragas_cache 文件夹
cache = DiskCacheBackend()

# 2. 初始化大模型客户端

#from google import genai
#client = genai.Client()

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
gemini_llm = ChatGoogleGenerativeAI(model=llm_model)
gemini_embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)

# 3. 构建工厂类，并使用缓存

# generator_llm = llm_factory(provider="google", model=llm_model, client=client, cache=cache)
# embeddings = embedding_factory(provider="google", model=embedding_model, client=client, cache=cache)
# from ragas.embeddings import GoogleEmbeddings
# embeddings = GoogleEmbeddings(client=client, model=embedding_model, cache=cache)

from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
generator_llm = LangchainLLMWrapper(gemini_llm, cache=cache)
embeddings = LangchainEmbeddingsWrapper(gemini_embeddings, cache=cache)

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
