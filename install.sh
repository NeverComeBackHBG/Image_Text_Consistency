#!/bin/bash
# AutoDL环境安装脚本
# 使用方法: bash install.sh

echo "=========================================="
echo "   图文相似度分析 - 环境安装脚本"
echo "=========================================="

# 1. 安装Ollama
echo "[1/5] 检查Ollama..."
if command -v ollama &> /dev/null; then
    echo "  ✓ Ollama已安装"
else
    echo "[1/5] 安装Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo "  ✓ Ollama安装完成"
fi

# 2. 启动Ollama服务
echo "[2/5] 启动Ollama服务..."
ollama serve &
sleep 3
echo "  ✓ Ollama服务已启动"

# 3. 下载Qwen2.5-VL模型
echo "[3/5] 下载Qwen2.5-VL模型（约15GB）..."
ollama pull qwen2.5-vl:7b-instruct
echo "  ✓ 模型下载完成"

# 4. 验证Ollama
echo "[4/5] 验证Ollama..."
ollama list

# 5. 安装Python依赖
echo "[5/5] 安装Python依赖..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "   环境安装完成！"
echo "=========================================="
echo ""
echo "下一步："
echo "  1. 上传数据到 data/input/ 目录"
echo "  2. 运行: python main.py --input data/input/你的数据.json --output data/output/result.json"
echo ""
