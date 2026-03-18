"""
单条数据测试脚本
用于快速测试单条图文数据的处理效果
"""

import json
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.pipeline import Pipeline


def test_single(text: str, image_path: str):
    """测试单条数据"""
    print("=" * 60)
    print("单条数据测试")
    print("=" * 60)
    print(f"\n文本: {text}")
    print(f"图像: {image_path}")

    # 初始化流程
    pipeline = Pipeline()

    # 处理
    result = pipeline.process_single(
        text=text,
        image_path=image_path,
        include_details=True
    )

    # 输出结果
    print("\n" + "=" * 60)
    print("处理结果")
    print("=" * 60)

    if result.get('success'):
        print(f"\n✓ 处理成功")
        print(f"  名词相似度: {result['noun_similarity']:.4f}")
        print(f"  形容词相似度: {result['adjective_similarity']:.4f}")
        print(f"  平均相似度: {result['average_similarity']:.4f}")
        print(f"  处理时间: {result['processing_time']:.2f}秒")

        if 'details' in result:
            print("\n详细信息:")
            print(f"  文本名词: {result['details']['text_nouns']}")
            print(f"  文本形容词: {result['details']['text_adjectives']}")
            print(f"  图像名词: {result['details']['image_nouns']}")
            print(f"  图像形容词: {result['details']['image_adjectives']}")
    else:
        print(f"\n✗ 处理失败: {result.get('error')}")

    return result


if __name__ == "__main__":
    # 默认测试数据
    default_text = "这家酒店位于繁华的市中心，装修豪华，服务周到，房间干净整洁"
    default_image = "https://via.placeholder.com/400x300.jpg"

    # 可以通过命令行参数指定
    if len(sys.argv) > 1:
        text = sys.argv[1]
    else:
        text = default_text

    if len(sys.argv) > 2:
        image = sys.argv[2]
    else:
        image = default_image

    test_single(text, image)
