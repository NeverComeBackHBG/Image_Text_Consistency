# 云端部署指南

本文档介绍如何将项目部署到云端GPU算力平台（如AutoDL）运行。

## 部署流程概览

```
本地开发 → GitHub → 云端服务器 → 运行 → 下载结果
```

---

## 方式一：AutoDL + Ollama镜像（推荐 ✅）

AutoDL有预装Ollama和Qwen2.5-VL的社区镜像，直接使用无需手动下载模型。

### 步骤1：注册AutoDL

1. 访问 https://www.autodl.com/
2. 完成注册和实名认证
3. 充值或购买算力积分

### 步骤2：租用GPU服务器

1. 登录AutoDL控制台
2. 点击"算力市场"
2. 选择服务器配置：
   - **RTX 4090**（24GB显存，约¥3/时）⭐推荐
   - **RTX 3090**（24GB显存，约¥2/时）
3. **关键步骤**：选择社区镜像
   - 在镜像搜索中输入：`qwen2.5-vl` 或 `ollama`
   - 选择包含 Ollama + Qwen2.5-VL 的镜像
   - 例如：`ubuntu22.04-cudnn11.8-vtorch2.3.0-ollama-qwen2.5vl`
4. 启动服务器

### 步骤3：部署项目

在AutoDL的Jupyter终端中执行：

```bash
# 1. 克隆项目
git clone https://github.com/你的用户名/你的仓库名.git
cd 你的仓库名

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 验证Ollama是否可用
ollama list
# 应该能看到 qwen2.5-vl:7b-instruct

# 5. 启动Ollama服务（如果未自动启动）
ollama serve &
```

### 步骤4：上传数据

将JSON数据文件上传到服务器的 `data/input/` 目录

### 步骤5：运行

```bash
# 批量处理
python main.py --input data/input/your_data.json --output data/output/result.json

# 或使用后台运行（长时间任务）
nohup python main.py --input data/input/your_data.json --output data/output/result.json > output.log 2>&1 &
```

### 步骤6：下载结果

处理完成后，从 `data/output/` 目录下载结果文件

---

## 方式二：AutoDL + 从零安装

如果社区镜像不可用，可以手动安装Ollama。

### 步骤1-2：同方式一

### 步骤3：手动安装Ollama

```bash
# 1. 安装Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. 启动Ollama服务
ollama serve &

# 3. 下载Qwen2.5-VL模型（约15GB）
ollama pull qwen2.5-vl:7b-instruct
```

### 后续步骤：同方式一

---

## 方式三：使用 OpenRouter API（可选）

如果不想用Ollama，可以使用云端API。

### 注册OpenRouter

1. 访问 https://openrouter.ai/
2. 注册账号
3. 在设置中获取 API Key

### 配置

编辑 `config/config.yaml`：

```yaml
models:
  vision:
    provider: "openrouter"
    model: "qwen/qwen2.5-vl-7b-instruct"
    base_url: "https://openrouter.ai/api/v1"
```

设置环境变量：

```bash
export OPENROUTER_API_KEY="sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### 运行

```bash
python main.py --input data/input/your_data.json --output data/output/result.json
```

### 费用估算

| 模型 | 每次调用费用 | 1000条费用 |
|------|-------------|------------|
| qwen2.5-vl-7b | ~$0.0015 | ~$1.5 |
| qwen2.5-vl-3b | ~$0.0008 | ~$0.8 |

---

## 性能与成本

### 1000条数据处理估算

| 配置 | 预计时间 | 预计费用 |
|------|---------|----------|
| RTX 4090 + Ollama | ~1.5小时 | ~¥4-5 |
| RTX 3090 + Ollama | ~2小时 | ~¥4 |
| OpenRouter API | ~30分钟 | ~$1.5 |

### 省钱技巧

1. 使用更小的模型（qwen2.5-vl:3b 更快更省显存）
2. 处理完数据后立即停止服务器
3. 错峰运行

---

## 故障排查

### 问题1：Ollama连接失败

**解决方案：**
```bash
# 检查Ollama是否运行
ps aux | grep ollama

# 手动启动
ollama serve
```

### 问题2：OOM（显存不足）

**解决方案：**
- 使用更小的模型：`qwen2.5-vl:3b-instruct`
- 或使用3B配置：`ollama pull qwen2.5-vl:3b-instruct`

### 问题3：图片加载失败

**检查：**
- 图片URL是否可访问
- 网络是否正常

### 问题4：模型未找到

**解决方案：**
```bash
# 查看已安装模型
ollama list

# 下载模型
ollama pull qwen2.5-vl:7b-instruct
```

---

## 监控与日志

### 查看运行日志

```bash
# 实时查看日志
tail -f output.log

# 查看错误
grep -i error output.log
```

### 监控GPU使用

```bash
nvidia-smi -l 1  # 每秒刷新
```

---

## 相关链接

- [AutoDL官网](https://www.autodl.com/)
- [Ollama官网](https://ollama.com/)
- [Qwen2.5-VL模型](https://ollama.com/library/qwen2.5-vl)
- [BGE-M3模型](https://huggingface.co/BAAI/bge-m3)
