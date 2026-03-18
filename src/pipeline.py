"""
主处理流程
整合文本处理、图像处理和相似度计算
"""

import yaml
from typing import Dict, List
from pathlib import Path
import json
import time
import os

from .text_processor import TextProcessor
from .image_processor import ImageProcessor
from .vectorizer import Vectorizer
from .similarity import SimilarityCalculator


class Pipeline:
    """图文相似度分析流程"""

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化处理流程

        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        config_file = Path(__file__).parent.parent / config_path
        if not config_file.exists():
            config_file = Path(config_path)

        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # 初始化各模块
        print("=" * 50)
        print("初始化处理模块...")
        print("=" * 50)

        # 获取缓存目录配置
        hanlp_cache = self.config.get('hanlp', {}).get('cache_dir', './models/hanlp')
        bge_cache = self.config.get('bge', {}).get('cache_dir', './models/bge')

        # 文本处理器
        self.text_processor = TextProcessor(
            pretrained=self.config['hanlp']['pretrained'],
            cache_dir=hanlp_cache
        )
        # 预加载HanLP
        self.text_processor.load()

        # 图像处理器
        vision_config = self.config['models']['vision']
        vision_cache = vision_config.get('cache_dir', './models')

        self.image_processor = ImageProcessor(
            provider=vision_config['provider'],
            model=vision_config['model'],
            base_url=vision_config.get('base_url', 'http://localhost:11434'),
            api_key=vision_config.get('api_key'),
            cache_dir=vision_cache
        )

        # 向量化器
        self.vectorizer = Vectorizer(
            model_name=self.config['models']['bge_model'],
            cache_dir=bge_cache
        )
        # 预加载BGE-M3
        self.vectorizer.load()

        # 相似度计算器
        self.similarity_calculator = SimilarityCalculator()

        print("\n所有模块初始化完成！")
        print("=" * 50)

    def process_single(
        self,
        text: str,
        image_path: str,
        include_details: bool = True
    ) -> Dict:
        """
        处理单条数据

        Args:
            text: 笔记文本
            image_path: 图像路径或URL
            include_details: 是否包含详细信息

        Returns:
            处理结果
        """
        start_time = time.time()

        try:
            # 1. 文本处理 - 抽取名词和形容词
            text_result = self.text_processor.process(text)

            # 2. 图像处理 - 提取图像中的名词和形容词
            image_result = self.image_processor.process(image_path)

            # 3. 相似度计算
            similarity_result = self.similarity_calculator.compute_similarities(
                text_nouns=text_result['noun_string'],
                text_adjectives=text_result['adjective_string'],
                image_nouns=image_result['noun_string'],
                image_adjectives=image_result['adjective_string'],
                vectorizer=self.vectorizer
            )

            # 组装结果
            result = {
                "success": True,
                "noun_similarity": similarity_result['noun_similarity'],
                "adjective_similarity": similarity_result['adjective_similarity'],
                "average_similarity": similarity_result['average_similarity'],
                "processing_time": round(time.time() - start_time, 2)
            }

            # 添加详细信息
            if include_details and self.config['output'].get('include_details', True):
                result["details"] = {
                    "text_nouns": text_result['nouns'],
                    "text_adjectives": text_result['adjectives'],
                    "image_nouns": image_result['nouns'],
                    "image_adjectives": image_result['adjectives']
                }

            return result

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "processing_time": round(time.time() - start_time, 2)
            }

    def process_batch(
        self,
        data: List[Dict],
        include_details: bool = False
    ) -> List[Dict]:
        """
        批量处理数据

        Args:
            data: 数据列表，每项包含 text 和 image 字段
            include_details: 是否包含详细信息

        Returns:
            处理结果列表
        """
        results = []

        total = len(data)
        print(f"\n开始批量处理，共 {total} 条数据")
        print("=" * 50)

        for i, item in enumerate(data):
            print(f"\n[{i+1}/{total}] 正在处理...")

            # 获取文本和图片字段
            text = item.get('text', item.get('content', ''))
            image_path = item.get('image', item.get('image_url', ''))

            result = self.process_single(
                text=text,
                image_path=image_path,
                include_details=include_details
            )

            # 保留原始数据标识
            if 'id' in item:
                result['id'] = item['id']

            results.append(result)

            # 打印进度
            if result.get('success'):
                print(f"  ✓ 名词相似度: {result['noun_similarity']:.4f}")
                print(f"  ✓ 形容词相似度: {result['adjective_similarity']:.4f}")
            else:
                print(f"  ✗ 处理失败: {result.get('error')}")

        # 统计
        success_count = sum(1 for r in results if r.get('success'))
        print("\n" + "=" * 50)
        print(f"批量处理完成: {success_count}/{total} 成功")

        # 计算平均耗时
        total_time = sum(r.get('processing_time', 0) for r in results)
        if success_count > 0:
            print(f"平均处理时间: {total_time/success_count:.2f}秒/条")

        return results


def load_data_from_json(json_path: str) -> List[Dict]:
    """从JSON文件加载数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_results_to_json(results: List[Dict], output_path: str):
    """保存结果到JSON文件"""
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def test():
    """测试函数"""
    pipeline = Pipeline()

    # 测试单条
    result = pipeline.process_single(
        text="这家酒店位于市中心，装修豪华，服务周到",
        image_path="https://via.placeholder.com/400x300.jpg"  # 示例图片
    )

    print("\n单条测试结果:")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    test()
