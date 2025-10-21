"""Minimal Retrieval-Augmented Generation (RAG) example using PyTorch.

This module demonstrates how Clash Verge Rev specific knowledge can be retrieved
with a TF-IDF retriever implemented on top of PyTorch tensors.  The retrieved
context is then stitched into a lightweight, template driven response for the
query.  The focus is on showing how to perform the similarity math with PyTorch
rather than on building a production ready pipeline.

中文说明：
    本示例演示如何使用 PyTorch 构建一个最小化的检索增强生成（RAG）流程。
    主要步骤包括：
        1. 预处理知识库文档并构建 TF-IDF 向量。
        2. 使用余弦相似度在向量空间中检索与查询最相关的文档。
        3. 将检索结果组合成一个简单的中文描述回答。
    代码的重点在于展示如何用 PyTorch 进行向量运算，而非搭建完整的生产级系统。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F


# 常见的英文停用词集合，用于在分词后过滤掉对语义贡献不大的词汇。
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
}


@dataclass
class KnowledgeBaseEntry:
    """Represents a single knowledge base document."""

    # 中文说明：表示知识库中的一条文档，包含唯一 ID、正文内容以及可选的元数据。 

    id: str
    content: str
    metadata: Dict[str, str] | None = None


@dataclass
class RetrievedDocument(KnowledgeBaseEntry):
    """Knowledge base entry with a similarity score."""

    # 中文说明：在检索阶段返回的文档，除了原始信息之外还会附带相似度得分。 

    score: float = 0.0


@dataclass
class RagResult:
    """Container for the output of the RAG pipeline."""

    # 中文说明：RAG 流程的最终结果，记录查询、检索到的文档列表以及生成的回答。 

    query: str
    retrieved_documents: List[RetrievedDocument]
    answer: str


# 预置的 Clash Verge Rev 知识库，每一条都是一段精简说明。
# 中文说明：实际项目中可以替换为数据库或向量索引，此处仅为了演示。
KNOWLEDGE_BASE: Sequence[KnowledgeBaseEntry] = (
    KnowledgeBaseEntry(
        id="tauri-integration",
        content=(
            "Clash Verge Rev packages its React interface with the Tauri framework. "
            "The desktop shell delegates privileged tasks, such as proxy control, "
            "filesystem access and auto updates, to the Rust backend while keeping "
            "the UI lightweight."
        ),
        metadata={
            "title": "Tauri integration",
            "url": "https://github.com/clash-verge-rev/clash-verge-rev",
        },
    ),
    KnowledgeBaseEntry(
        id="proxy-profiles",
        content=(
            "Users can manage multiple proxy profiles. Each profile wraps rule sets, "
            "proxy groups and tun settings so switching between providers is instant "
            "without editing YAML manually."
        ),
        metadata={"title": "Proxy profile management"},
    ),
    KnowledgeBaseEntry(
        id="updater",
        content=(
            "The application bundles an updater that can pull stable, alpha or "
            "autobuild releases. It compares semantic versions and downloads the "
            "correct package for the platform."
        ),
        metadata={"title": "Multi-channel updater"},
    ),
    KnowledgeBaseEntry(
        id="webdav-sync",
        content=(
            "Configuration files can be synchronized over WebDAV. When enabled, the "
            "client pushes changes after they are saved and pulls remote edits to "
            "keep devices in sync."
        ),
        metadata={"title": "WebDAV sync"},
    ),
)


def tokenize(text: str) -> List[str]:
    """Splits text into lowercase tokens while removing stop words."""

    # 中文说明：将文本转换为小写后按字母数字切分，并去除停用词。
    # 这里使用最朴素的方式，仅适用于英文示例。若想支持中文需要换更合适的分词器。

    return [
        token
        for token in "".join(
            char.lower() if char.isalnum() else " " for char in text
        ).split()
        if token and token not in STOP_WORDS
    ]


class TfidfRetriever:
    """Simple TF-IDF retriever backed by PyTorch tensors."""

    # 中文说明：此类负责基于 TF-IDF 的向量表示来完成检索工作。
    # 初始化时会预计算词汇表、IDF 权重和每篇文档的向量表示。

    def __init__(self, documents: Sequence[KnowledgeBaseEntry]):
        self._documents = list(documents)
        # 中文说明：对所有文档进行分词，后续步骤会在这些分词结果上进行统计。
        self._document_tokens = [tokenize(entry.content) for entry in self._documents]
        # 中文说明：构建词汇表，将每个词映射到一个索引位置。
        self._token_to_index = self._build_vocabulary(self._document_tokens)
        # 中文说明：计算每个词的 IDF（逆文档频率），衡量其区分度。
        self._idf = self._compute_idf(self._document_tokens)
        # 中文说明：将每篇文档转换为 TF-IDF 向量，并在初始化阶段堆叠成张量。
        self._document_vectors = torch.stack(
            [self._vectorize(tokens) for tokens in self._document_tokens]
        )

    def _build_vocabulary(self, documents: Sequence[Sequence[str]]) -> Dict[str, int]:
        tokens = sorted({token for doc in documents for token in doc})
        # 中文说明：通过集合去重的方式收集所有词，并保持有序以保证索引稳定。
        return {token: index for index, token in enumerate(tokens)}

    def _compute_idf(self, documents: Sequence[Sequence[str]]) -> torch.Tensor:
        doc_count = len(documents)
        idf_counts = torch.zeros(len(self._token_to_index), dtype=torch.float32)
        for tokens in documents:
            unique_indices = {self._token_to_index[token] for token in tokens}
            for index in unique_indices:
                idf_counts[index] += 1
        # 中文说明：应用加一平滑避免分母为零，并用对数公式得到最终 IDF 值。
        return torch.log((doc_count + 1) / (idf_counts + 1)) + 1

    def _vectorize(self, tokens: Sequence[str]) -> torch.Tensor:
        vector = torch.zeros(len(self._token_to_index), dtype=torch.float32)
        for token in tokens:
            index = self._token_to_index.get(token)
            if index is not None:
                vector[index] += 1
        # 中文说明：TF（词频）采用最简单的词频统计并进行归一化。
        total_terms = vector.sum()
        if total_terms > 0:
            vector = vector / total_terms
        # 中文说明：将归一化后的 TF 与预先计算的 IDF 相乘得到 TF-IDF 向量。
        return vector * self._idf

    def retrieve(self, query: str, top_k: int = 2) -> List[RetrievedDocument]:
        # 中文说明：对查询语句执行同样的分词与向量化流程。
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        query_vector = self._vectorize(query_tokens)
        if torch.allclose(query_vector, torch.zeros_like(query_vector)):
            return []

        scores = F.cosine_similarity(
            query_vector.unsqueeze(0), self._document_vectors, dim=1, eps=1e-8
        )
        # 中文说明：使用余弦相似度衡量查询向量与每篇文档向量的相关性。
        ranked_indices = scores.argsort(descending=True)
        retrieved: List[RetrievedDocument] = []
        for index in ranked_indices[:top_k]:
            score = scores[index].item()
            if score <= 0:
                continue
            entry = self._documents[index]
            retrieved.append(
                RetrievedDocument(
                    id=entry.id,
                    content=entry.content,
                    metadata=entry.metadata,
                    score=score,
                )
            )
        # 中文说明：返回得分从高到低的前 top_k 个文档。
        return retrieved


def synthesize_answer(query: str, documents: Sequence[RetrievedDocument]) -> str:
    """Produces a naive answer by weaving the retrieved passages together."""

    # 中文说明：将检索结果整合成可读的回答。如果没有相关文档，则返回兜底提示。
    if not documents:
        return "I could not find any relevant information in the knowledge base."

    bullet_points = []
    for doc in documents:
        description = doc.metadata.get("title") if doc.metadata else None
        if description:
            bullet_points.append(f"- {description}: {doc.content}")
        else:
            bullet_points.append(f"- {doc.content}")
    context = "\n".join(bullet_points)
    # 中文说明：最终回答采用模板方式输出，将上下文信息逐条列出。
    return (
        f"Question: {query}\n\n"
        f"Here's what I found in the Clash Verge Rev knowledge base:\n{context}\n\n"
        "These passages should address the query above."
    )


def run_rag_pipeline(query: str, top_k: int = 2) -> RagResult:
    """Runs retrieval and answer synthesis for a given query."""

    # 中文说明：封装完整流程，先构建检索器、获取最相关的文档，再生成回答。
    retriever = TfidfRetriever(KNOWLEDGE_BASE)
    retrieved = retriever.retrieve(query, top_k=top_k)
    answer = synthesize_answer(query, retrieved)
    return RagResult(query=query, retrieved_documents=retrieved, answer=answer)


if __name__ == "__main__":
    example_query = "How does Clash Verge Rev integrate with Tauri and manage updates?"
    result = run_rag_pipeline(example_query, top_k=3)
    # 中文说明：打印生成的答案以及被检索出的文档和相似度得分。
    print(result.answer)
    print("\nRetrieved documents:")
    for doc in result.retrieved_documents:
        meta = f" ({doc.metadata['title']})" if doc.metadata and "title" in doc.metadata else ""
        print(f"* {doc.id}{meta} — score={doc.score:.3f}")
