"""
向量化模块
使用BGE-M3进行文本向量化
"""

from FlagEmbedding import FlagModel
from typing import List
import numpy as np
import os


class Vectorizer:
    """文本向量化器 - 使用BGE-M3"""

    def __init__(self, model_name: str = "BAAI/bge-m3", cache_dir: str = None):
        """
        初始化向量化器

        Args:
            model_name: BGE-M3模型名称
            cache_dir: 模型缓存目录（默认项目目录的models/bge）
        """
        self.model_name = model_name

        # 设置模型缓存目录
        if cache_dir:
            self.cache_dir = cache_dir
        else:
            self.cache_dir = "./models/bge"

        # 创建缓存目录
        os.makedirs(self.cache_dir, exist_ok=True)

        # 设置HuggingFace缓存目录
        os.environ['HF_HOME'] = self.cache_dir
        os.environ['TRANSFORMERS_CACHE'] = self.cache_dir

        self.model = None
        self._loaded = False

    def load(self):
        """加载BGE-M3模型（延迟加载）"""
        if self._loaded:
            return

        print(f"正在加载BGE-M3模型: {self.model_name}")
        print(f"模型缓存目录: {self.cache_dir}")
        print("首次运行需要下载模型，请耐心等待...")

        # 使用FP16加速，减少显存占用
        self.model = FlagModel(self.model_name, use_fp16=True)

        self._loaded = True
        print("BGE-M3模型加载完成")

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        批量编码文本

        Args:
            texts: 文本列表

        Returns:
            向量数组 (n, dim)
        """
        if not self._loaded:
            self.load()

        embeddings = self.model.encode(texts)
        return embeddings

    def encode_single(self, text: str) -> np.ndarray:
        """
        编码单个文本

        Args:
            text: 输入文本

        Returns:
            向量 (dim,)
        """
        if not self._loaded:
            self.load()

        embedding = self.model.encode([text])[0]
        return embedding

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的余弦相似度

        Args:
            text1: 文本1
            text2: 文本2

        Returns:
            余弦相似度 (0-1)
        """
        if not text1.strip() or not text2.strip():
            return 0.0

        emb1 = self.encode_single(text1)
        emb2 = self.encode_single(text2)

        # 余弦相似度
        cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

        # 确保在[0, 1]范围内
        cos_sim = max(0.0, min(1.0, cos_sim))

        return float(cos_sim)


def test():
    """测试函数"""
    vectorizer = Vectorizer()
    vectorizer.load()

    # 测试文本
    texts = [
        "美丽的蓝天白云",
        "晴朗的天空",
        "电脑显示器"
    ]

    embeddings = vectorizer.encode(texts)
    print(f"嵌入维度: {embeddings.shape}")

    # 测试相似度
    sim1 = vectorizer.compute_similarity("美丽的蓝天", "晴朗的天空")
    print(f"'美丽的蓝天' vs '晴朗的天空' 相似度: {sim1:.4f}")

    sim2 = vectorizer.compute_similarity("美丽的蓝天", "电脑")
    print(f"'美丽的蓝天' vs '电脑' 相似度: {sim2:.4f}")


if __name__ == "__main__":
    test()
