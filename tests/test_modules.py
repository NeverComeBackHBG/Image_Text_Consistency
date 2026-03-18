"""
单元测试模块
测试各个模块的功能
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_text_processor():
    """测试文本处理模块"""
    print("\n" + "=" * 50)
    print("测试: TextProcessor")
    print("=" * 50)

    from src.text_processor import TextProcessor

    processor = TextProcessor()
    processor.load()

    test_cases = [
        "这家酒店位于繁华的市中心，装修豪华，服务周到",
        "风景优美的古镇，蓝天白云，古色古香的建筑",
        "海鲜新鲜美味，价格实惠，环境优雅"
    ]

    for text in test_cases:
        result = processor.process(text)
        print(f"\n原文: {text}")
        print(f"名词: {result['nouns']}")
        print(f"形容词: {result['adjectives']}")

    print("\n✓ TextProcessor 测试通过")


def test_vectorizer():
    """测试向量化模块"""
    print("\n" + "=" * 50)
    print("测试: Vectorizer")
    print("=" * 50)

    from src.vectorizer import Vectorizer

    vectorizer = Vectorizer()
    vectorizer.load()

    # 测试编码
    texts = ["美丽的蓝天", "晴朗的天空", "电脑"]
    embeddings = vectorizer.encode(texts)
    print(f"嵌入维度: {embeddings.shape}")

    # 测试相似度
    sim1 = vectorizer.compute_similarity("美丽的蓝天", "晴朗的天空")
    sim2 = vectorizer.compute_similarity("美丽的蓝天", "电脑")

    print(f"'美丽的蓝天' vs '晴朗的天空': {sim1:.4f}")
    print(f"'美丽的蓝天' vs '电脑': {sim2:.4f}")

    assert sim1 > sim2, "相似度计算异常"
    print("\n✓ Vectorizer 测试通过")


def test_similarity():
    """测试相似度计算"""
    print("\n" + "=" * 50)
    print("测试: SimilarityCalculator")
    print("=" * 50)

    from src.vectorizer import Vectorizer
    from src.similarity import SimilarityCalculator

    vectorizer = Vectorizer()
    vectorizer.load()

    calculator = SimilarityCalculator()

    result = calculator.compute_similarities(
        text_nouns="天空 白云 阳光",
        text_adjectives="晴朗 温暖",
        image_nouns="天空 白云 草地",
        image_adjectives="晴朗 清新",
        vectorizer=vectorizer
    )

    print(f"名词相似度: {result['noun_similarity']}")
    print(f"形容词相似度: {result['adjective_similarity']}")
    print(f"平均相似度: {result['average_similarity']}")

    print("\n✓ SimilarityCalculator 测试通过")


def test_image_processor_parse():
    """测试图像处理器解析功能"""
    print("\n" + "=" * 50)
    print("测试: ImageProcessor (解析)")
    print("=" * 50)

    from src.image_processor import ImageProcessor

    processor = ImageProcessor()

    # 测试各种JSON格式
    test_jsons = [
        '{"nouns": ["天空", "城市"], "adjectives": ["美丽", "繁华"]}',
        '```json\n{"nouns": ["天空"], "adjectives": ["晴朗"]}\n```',
        '{"nouns": ["建筑"], "adjectives": ["古老"]}'
    ]

    for json_str in test_jsons:
        nouns, adjectives = processor._parse_response(json_str)
        print(f"输入: {json_str[:50]}...")
        print(f"  名词: {nouns}, 形容词: {adjectives}")

    print("\n✓ ImageProcessor 解析测试通过")


def run_all_tests():
    """运行所有测试"""
    print("=" * 50)
    print("开始运行单元测试")
    print("=" * 50)

    try:
        test_text_processor()
        test_vectorizer()
        test_similarity()
        test_image_processor_parse()

        print("\n" + "=" * 50)
        print("✓ 所有测试通过!")
        print("=" * 50)

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
