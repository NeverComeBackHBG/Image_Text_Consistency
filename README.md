# 图文一致性分析 (Image-Text Consistency Analysis)

分析文本描述与图像内容的一致性程度。

## 功能

- 使用 **Qwen2.5-VL** 理解图像内容
- 使用 **BGE-M3** 进行文本向量化
- 计算图文相似度（名词 + 形容词）
- 支持 GPU 加速

## 环境要求

- Python 3.9+
- CUDA 12.1+
- GPU 显存 ≥ 16GB（推荐 24GB）
- 硬盘空间 ≥ 30GB（模型缓存）

## 快速开始

### 1. 安装依赖

```bash
# 方式一：直接安装
pip install -r requirements.txt

# 方式二：运行环境搭建脚本
python setup_environment.py
```

### 2. 下载模型

模型会自动下载到 `./models` 目录（首次运行时）。

如需手动下载：
```bash
# Qwen2.5-VL-7B
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct

# BGE-M3
huggingface-cli download BAAI/bge-m3
```

### 3. 运行 Demo

```bash
# JupyterLab
jupyter lab demo.ipynb
```

### 4. 配置国内镜像（推荐）

首次运行前设置环境变量：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1
```

## 使用方法

### Python API

```python
from src.pipeline import ImageTextPipeline

# 创建 Pipeline
pipeline = ImageTextPipeline(
    provider="huggingface",  # 使用 HuggingFace 模型
    cache_dir="./models"
)

# 分析图文一致性
result = pipeline.analyze(
    text="酒店位于市中心，装修豪华",
    image_path="https://example.com/hotel.jpg"
)

print(f"相似度: {result['average_similarity']:.4f}")
```

### 命令行

```bash
python run.py --text "酒店位于市中心" --image_path "./test.jpg"
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| provider | 模型来源 | "huggingface" |
| cache_dir | 模型缓存目录 | "./models" |
| use_quantization | 是否使用量化 | True |
| quantization_bits | 量化位数 (4/8) | 8 |

### Provider 选项

- `huggingface`: 从 HuggingFace 下载模型（推荐）
- `ollama`: 使用本地 Ollama 服务
- `openrouter`: 使用 OpenRouter API（需 API Key）

## 项目结构

```
Image_Text_Consistency/
├── src/
│   ├── image_processor.py   # 图像处理（Qwen2.5-VL）
│   ├── text_processor.py   # 文本处理（jieba）
│   ├── vectorizer.py       # 向量化（BGE-M3）
│   ├── similarity.py       # 相似度计算
│   └── pipeline.py         # 完整流程
├── config/
│   └── config.yaml         # 配置文件
├── data/
│   ├── input/              # 输入数据
│   └── output/             # 输出结果
├── models/                 # 模型缓存
├── demo.ipynb              # Jupyter Demo
├── run.py                  # 命令行入口
├── requirements.txt        # 依赖清单
└── setup_environment.py   # 环境搭建脚本
```

## 显存优化

如遇显存不足：

1. **使用 8-bit 量化**（默认）
```python
pipeline = ImageTextPipeline(use_quantization=True, quantization_bits=8)
```

2. **使用 4-bit 量化**（更省显存）
```python
pipeline = ImageTextPipeline(use_quantization=True, quantization_bits=4)
```

3. **关闭量化**（需要更多显存）
```python
pipeline = ImageTextPipeline(use_quantization=False)
```

4. **处理完后清理显存**
```python
import torch
torch.cuda.empty_cache()
```

## 常见问题

### Q: 模型下载很慢
A: 设置国内镜像：
```python
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

### Q: 显存不足
A: 
- 使用量化模式（默认已开启）
- 减少 batch_size
- 清理其他占用显存的进程

### Q: 图像加载失败
A: 
- 检查 URL 是否可访问
- 尝试下载到本地后使用本地路径

## License

MIT License
