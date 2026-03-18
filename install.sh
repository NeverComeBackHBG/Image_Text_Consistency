#!/bin/bash
# AutoDL环境安装脚本（社区镜像版）
# 假设：社区镜像已预装Ollama和Qwen2.5-VL模型

echo "=========================================="
echo "   图文相似度分析 - 环境安装脚本"
echo "=========================================="

# 1. 检查Ollama
echo "[1/4] 检查Ollama..."
if command -v ollama &> /dev/null; then
    echo "  ✓ Ollama已安装"
    ollama list
else
    echo "  ✗ Ollama未安装，请使用社区镜像"
fi

# 2. 检查CUDA
echo "[2/4] 检查CUDA..."
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# 3. 安装Python依赖
echo "[3/4] 安装Python依赖..."
pip install -r requirements.txt

# 4. 确认模型可用
echo "[4/4] 检查可用模型..."
ollama list

echo ""
echo "=========================================="
echo "   环境安装完成！"
echo "=========================================="
echo ""
echo "下一步："
echo "  1. 上传数据到 data/input/ 目录"
echo "  2. 运行: python main.py --input data/input/你的数据.json --output data/output/result.json"
echo ""
