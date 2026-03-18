#!/usr/bin/env python3
"""
环境搭建脚本 - RTX 5090 新环境
自动安装所有依赖和下载模型
"""

import os
import sys
import subprocess

def run_command(cmd, description=""):
    """运行命令并显示进度"""
    print(f"\n{'='*60}")
    if description:
        print(f"📦 {description}")
    print(f"执行: {cmd}")
    print('='*60)
    
    result = subprocess.run(cmd, shell=True, capture_output=False)
    return result.returncode == 0

def check_cuda():
    """检查CUDA环境"""
    print("\n🔍 检查 CUDA 环境...")
    try:
        import torch
        print(f"   PyTorch 版本: {torch.__version__}")
        print(f"   CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU 数量: {torch.cuda.device_count()}")
            print(f"   GPU 型号: {torch.cuda.get_device_name(0)}")
            print(f"   显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        return True
    except ImportError:
        print("   ❌ PyTorch 未安装")
        return False

def install_dependencies():
    """安装依赖"""
    print("\n📦 安装 Python 依赖...")
    
    # 基础依赖
    deps = [
        "pip install --upgrade pip",
        "pip install PyYAML requests numpy pillow jieba",
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
        "pip install transformers>=4.40.0 accelerate bitsandbytes qwen-vl-utils",
        "pip install FlagEmbedding",
    ]
    
    for dep in deps:
        if not run_command(dep, f"安装 {dep.split('pip install')[-1].strip().split()[0]}"):
            print(f"❌ 安装失败: {dep}")
            return False
    
    return True

def setup_huggingface_mirror():
    """配置 HuggingFace 镜像"""
    print("\n🔧 配置 HuggingFace 镜像...")
    
    # 设置环境变量
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
    
    # 创建配置文件
    hf_home = os.path.expanduser("~/.cache/huggingface")
    os.makedirs(hf_home, exist_ok=True)
    
    config_path = os.path.join(hf_home, "hub")
    os.makedirs(config_path, exist_ok=True)
    
    print("   ✅ HuggingFace 镜像配置完成")
    return True

def download_models():
    """预下载模型"""
    print("\n📥 预下载模型...")
    
    print("\n   1/2 下载 Qwen2.5-VL-7B 模型...")
    cmd1 = '''
python3 -c "
from transformers import AutoProcessor, AutoModelForVision2Seq
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
print('开始下载 Qwen2.5-VL-7B-Instruct...')
processor = AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct', trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct', trust_remote_code=True)
print('下载完成!')
"
'''
    run_command(cmd1, "下载 Qwen2.5-VL 模型")
    
    print("\n   2/2 下载 BGE-M3 模型...")
    cmd2 = '''
python3 -c "
from FlagEmbedding import BGEM3FlagModel
print('开始下载 BGE-M3...')
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
print('下载完成!')
"
'''
    run_command(cmd2, "下载 BGE-M3 模型")
    
    return True

def test_installation():
    """测试安装"""
    print("\n🧪 测试安装...")
    
    try:
        import torch
        print(f"   ✅ PyTorch: {torch.__version__}")
        
        from transformers import AutoProcessor
        print(f"   ✅ Transformers")
        
        from FlagEmbedding import BGEM3FlagModel
        print(f"   ✅ FlagEmbedding")
        
        import jieba
        print(f"   ✅ Jieba")
        
        if torch.cuda.is_available():
            print(f"   ✅ CUDA: {torch.cuda.get_device_name(0)}")
        
        return True
    except Exception as e:
        print(f"   ❌ 测试失败: {e}")
        return False

def main():
    """主函数"""
    print("="*60)
    print("🚀 图文相似度分析环境搭建")
    print("   目标: RTX 5090 (24GB+ VRAM)")
    print("="*60)
    
    # 检查 CUDA
    check_cuda()
    
    # 安装依赖
    if not install_dependencies():
        print("\n❌ 依赖安装失败")
        return 1
    
    # 配置镜像
    setup_huggingface_mirror()
    
    # 下载模型（可选）
    print("\n❓ 是否现在下载模型？(y/n)")
    print("   建议首次运行输入 y，后续可跳过")
    choice = input("   > ").strip().lower()
    if choice == 'y':
        download_models()
    
    # 测试
    if test_installation():
        print("\n" + "="*60)
        print("🎉 环境搭建完成!")
        print("="*60)
        print("\n📝 下一步:")
        print("   1. 运行 demo.ipynb 测试")
        print("   2. 修改输入数据")
        print("   3. 执行分析")
        return 0
    else:
        print("\n❌ 环境测试失败，请检查错误信息")
        return 1

if __name__ == "__main__":
    sys.exit(main())
