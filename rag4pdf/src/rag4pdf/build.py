import glob
import logging
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import default_transforms, apply_transforms
from ragas.run_config import RunConfig

logger = logging.getLogger(__name__)

def load_markdown_documents(pattern: str) -> List[Document]:
    """
    使用 Glob 加载 Markdown 文件并转换为 Langchain Document。
    """
    documents = []
    files = glob.glob(pattern)

    if not files:
        logger.warning(f"No files found matching pattern: {pattern}")
        return []

    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        documents.append(Document(page_content=content, metadata={"source": file_path}))

    logger.info(f"Loaded {len(documents)} documents from {pattern}")
    return documents


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 100
) -> List[Document]:
    """
    使用 RecursiveCharacterTextSplitter 对文档进行切分。
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    chunked_documents = text_splitter.split_documents(documents)
    logger.info(f"Split {len(documents)} documents into {len(chunked_documents)} chunks")
    return chunked_documents


def build_knowledge_graph(
    documents: List[Document],
    llm,
    embeddings,
    output_path: str,
    run_config: Optional[RunConfig] = None
) -> KnowledgeGraph:
    """
    封装 Ragas 知识图谱生成与保存逻辑。
    """
    if run_config is None:
        run_config = RunConfig(
            max_workers=5,
            max_retries=10,
            timeout=180
        )

    # 1. 将 Langchain Documents 转换为 Ragas Nodes，并初始化知识图谱
    nodes = [Node(
        type=NodeType.DOCUMENT,
        properties={"page_content": doc.page_content}
    ) for doc in documents]

    kg = KnowledgeGraph(nodes=nodes)

    # 2. 配置并运行变换 (Extractors, Splitters, etc.)
    transformers = default_transforms(documents, llm, embeddings)

    # 开始构建
    apply_transforms(kg, transformers, run_config=run_config)

    # 3. 保存图谱到本地
    kg.save(output_path)
    logger.info(f"Knowledge graph saved to {output_path}")

    return kg
