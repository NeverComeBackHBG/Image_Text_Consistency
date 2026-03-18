"""
批量处理脚本
用于处理大量图文数据
"""

import argparse
import json
import sys
import time
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.pipeline import Pipeline, load_data_from_json, save_results_to_json


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="批量处理图文相似度分析"
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        default="data/input/data.json",
        help="输入JSON文件路径"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/output/result.json",
        help="输出JSON文件路径"
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/config.yaml",
        help="配置文件路径"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="限制处理数量，0表示处理全部"
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    print("=" * 60)
    print("           批量图文相似度分析")
    print("=" * 60)

    # 加载数据
    print(f"\n加载数据: {args.input}")
    data = load_data_from_json(args.input)

    # 限制数量
    if args.limit > 0:
        data = data[:args.limit]
        print(f"限制处理数量: {len(data)}")

    print(f"共 {len(data)} 条数据")

    # 记录开始时间
    start_time = time.time()

    # 初始化流程
    print(f"\n初始化处理流程...")
    pipeline = Pipeline(config_path=args.config)

    # 批量处理
    print("\n开始处理...")
    results = pipeline.process_batch(data, include_details=False)

    # 统计
    end_time = time.time()
    total_time = end_time - start_time
    success_count = sum(1 for r in results if r.get('success'))

    # 保存结果
    save_results_to_json(results, args.output)
    print(f"\n结果已保存: {args.output}")

    # 打印统计
    print("\n" + "=" * 60)
    print("处理统计")
    print("=" * 60)
    print(f"  总数量: {len(results)}")
    print(f"  成功: {success_count}")
    print(f"  失败: {len(results) - success_count}")
    print(f"  总耗时: {total_time:.2f}秒")

    if success_count > 0:
        print(f"  平均耗时: {total_time/success_count:.2f}秒/条")

    # 导出核心结果（仅包含相似度）
    simple_results = []
    for r in results:
        simple_results.append({
            "id": r.get("id", ""),
            "noun_similarity": r.get("noun_similarity", 0),
            "adjective_similarity": r.get("adjective_similarity", 0),
            "average_similarity": r.get("average_similarity", 0),
            "success": r.get("success", False)
        })

    simple_output = args.output.replace(".json", "_simple.json")
    save_results_to_json(simple_results, simple_output)
    print(f"\n简化结果已保存: {simple_output}")


if __name__ == "__main__":
    main()
