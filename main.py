"""
图文相似度分析 - 主程序入口
"""

import argparse
import json
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.pipeline import Pipeline, load_data_from_json, save_results_to_json


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="图文相似度分析 - 计算名词相似度和形容词相似度"
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="输入JSON文件路径"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="输出JSON文件路径"
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/config.yaml",
        help="配置文件路径"
    )

    parser.add_argument(
        "--details",
        action="store_true",
        help="是否包含详细信息（名词/形容词列表）"
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    print("=" * 60)
    print("           图文相似度分析系统")
    print("=" * 60)

    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        return

    # 加载数据
    print(f"\n加载数据: {args.input}")
    data = load_data_from_json(args.input)
    print(f"共 {len(data)} 条数据")

    # 初始化流程
    print(f"\n初始化处理流程: {args.config}")
    pipeline = Pipeline(config_path=args.config)

    # 批量处理
    print("\n开始处理...")
    results = pipeline.process_batch(data, include_details=args.details)

    # 保存结果
    save_results_to_json(results, args.output)
    print(f"\n结果已保存: {args.output}")

    # 统计
    success = sum(1 for r in results if r.get('success'))
    print(f"\n处理统计: {success}/{len(results)} 成功")

    # 打印示例结果
    if results and results[0].get('success'):
        print("\n" + "-" * 40)
        print("示例结果 (第一条):")
        print(f"  名词相似度: {results[0]['noun_similarity']}")
        print(f"  形容词相似度: {results[0]['adjective_similarity']}")
        print(f"  平均相似度: {results[0]['average_similarity']}")
        print("-" * 40)


if __name__ == "__main__":
    main()
