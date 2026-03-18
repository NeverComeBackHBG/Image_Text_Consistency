"""
相似度计算模块
计算名词相似度和形容词相似度
"""

import numpy as np
from typing import Dict


class SimilarityCalculator:
    """相似度计算器"""

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        计算余弦相似度

        Args:
            vec1: 向量1
            vec2: 向量2

        Returns:
            余弦相似度
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def compute_similarities(
        self,
        text_nouns: str,
        text_adjectives: str,
        image_nouns: str,
        image_adjectives: str,
        vectorizer
    ) -> Dict[str, float]:
        """
        计算名词相似度和形容词相似度

        Args:
            text_nouns: 文本中的名词（空格连接）
            text_adjectives: 文本中的形容词（空格连接）
            image_nouns: 图像中的名词（空格连接）
            image_adjectives: 图像中的形容词（空格连接）
            vectorizer: 向量化器实例

        Returns:
            {
                "noun_similarity": 0.0 - 1.0,
                "adjective_similarity": 0.0 - 1.0,
                "average_similarity": 0.0 - 1.0
            }
        """
        # 计算名词相似度
        if text_nouns.strip() and image_nouns.strip():
            noun_similarity = vectorizer.compute_similarity(text_nouns, image_nouns)
        else:
            noun_similarity = 0.0

        # 计算形容词相似度
        if text_adjectives.strip() and image_adjectives.strip():
            adjective_similarity = vectorizer.compute_similarity(
                text_adjectives, image_adjectives
            )
        else:
            adjective_similarity = 0.0

        # 计算平均相似度
        similarities = [noun_similarity, adjective_similarity]
        valid_sims = [s for s in similarities if s > 0]
        average_similarity = np.mean(valid_sims) if valid_sims else 0.0

        return {
            "noun_similarity": round(noun_similarity, 4),
            "adjective_similarity": round(adjective_similarity, 4),
            "average_similarity": round(average_similarity, 4)
        }


def test():
    """测试函数"""
    from vectorizer import Vectorizer

    vectorizer = Vectorizer()
    vectorizer.load()

    calculator = SimilarityCalculator()

    # 测试数据
    result = calculator.compute_similarities(
        text_nouns="天空 白云 阳光",
        text_adjectives="晴朗 温暖",
        image_nouns="天空 白云 草地",
        image_adjectives="晴朗 清新",
        vectorizer=vectorizer
    )

    print("=" * 50)
    print("相似度计算测试")
    print("=" * 50)
    print(f"  名词相似度: {result['noun_similarity']}")
    print(f"  形容词相似度: {result['adjective_similarity']}")
    print(f"  平均相似度: {result['average_similarity']}")


if __name__ == "__main__":
    test()
