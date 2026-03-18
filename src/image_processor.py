"""
图像处理模块
使用Qwen2.5-VL生成图像的关键词项（名词+形容词）
支持三种方式：HuggingFace（推荐）、Ollama、OpenRouter API
"""

import json
import os
import re
import uuid
from typing import Dict, List, Optional
from PIL import Image
import requests
import base64
import io


class ImageProcessor:
    """图像处理器 - 提取图像中的名词和形容词"""

    DEFAULT_PROMPT = """请分析这张图片，提取：
1. 图像中出现的所有实体/对象（名词），如天空、城市、人物、物品等
2. 描述这些实体/场景的氛围和特征的词（形容词），如美丽、拥挤、安静、热闹等

请严格按照以下JSON格式输出，不要有其他内容：
{
    "nouns": ["词1", "词2"],
    "adjectives": ["词1", "词2"]
}

只输出JSON，不要其他解释。"""

    # HuggingFace 模型名称
    DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
    
    # 国内镜像（可选）
    HF_MIRROR = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

    def __init__(
        self,
        provider: str = "huggingface",
        model: str = None,
        base_url: str = "http://localhost:11434",
        api_key: Optional[str] = None,
        cache_dir: str = "./models",
        temp_dir: str = "./temp_images",
        use_quantization: bool = True,
        quantization_bits: int = 8
    ):
        """
        初始化图像处理器

        Args:
            provider: "huggingface"（推荐）、"ollama"、"openrouter"
            model: 模型名称（huggingface用）
            base_url: API地址（Ollama/OpenRouter用）
            api_key: API密钥
            cache_dir: 模型缓存目录
            temp_dir: 临时图片下载目录
            use_quantization: 是否使用量化（省显存）
            quantization_bits: 量化位数（4或8）
        """
        self.provider = provider
        self.model_name = model or self.DEFAULT_MODEL
        self.base_url = base_url
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.cache_dir = cache_dir
        self.temp_dir = temp_dir
        self.use_quantization = use_quantization
        self.quantization_bits = quantization_bits
        self._loaded = False

        # 创建临时目录和缓存目录
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

    def load(self):
        """加载模型"""
        if self._loaded:
            return

        print(f"视觉模型配置: {self.provider} - {self.model_name}")

        if self.provider == "huggingface":
            self._load_huggingface_model()
        elif self.provider == "ollama":
            # Ollama 不需要预加载
            pass

        self._loaded = True
        
        # 预热模型（可选，加速第一次推理）
        print("模型预热中...")
        try:
            self._warmup()
            print("模型预热完成")
        except Exception as e:
            print(f"预热失败: {e}")
        
        print("视觉模型就绪")

    def _load_huggingface_model(self):
        """从HuggingFace加载Qwen2.5-VL模型"""
        import torch
        from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"正在加载模型: {self.model_name}")
        print(f"使用设备: {device}")
        
        # 设置国内镜像
        if self.HF_MIRROR:
            os.environ['HF_ENDPOINT'] = self.HF_MIRROR
            print(f"HuggingFace 镜像: {self.HF_MIRROR}")
        
        # 先清理显存
        if device == "cuda":
            import torch
            torch.cuda.empty_cache()
            print(f"加载前显存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

        # 加载 processor
        print("加载 Processor...")
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=True
        )
        
        # 加载模型
        if device == "cuda":
            if self.use_quantization:
                # 使用量化加载（省显存）
                bits = self.quantization_bits
                print(f"使用 {bits}-bit 量化加载...")
                
                if bits == 8:
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0,
                    )
                else:  # 4-bit
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                
                try:
                    torch.cuda.empty_cache()
                    self.model = AutoModelForVision2Seq.from_pretrained(
                        self.model_name,
                        quantization_config=quantization_config,
                        device_map="auto",
                        cache_dir=self.cache_dir,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                    )
                    print(f"加载后显存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                    print(f"{bits}-bit 量化加载成功!")
                except Exception as e:
                    print(f"量化加载失败: {e}")
                    print("尝试 bfloat16 原始精度...")
                    torch.cuda.empty_cache()
                    self.model = AutoModelForVision2Seq.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                        cache_dir=self.cache_dir,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                    )
                    print(f"bfloat16 加载成功! 显存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            else:
                # 不使用量化
                print("使用 bfloat16 原始精度加载...")
                torch.cuda.empty_cache()
                self.model = AutoModelForVision2Seq.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    cache_dir=self.cache_dir,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
                print(f"加载后显存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        else:
            # CPU 模式
            print("使用 CPU 模式...")
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                cache_dir=self.cache_dir,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            print("CPU 模式加载完成")

        print("Qwen2.5-VL 模型加载完成")

    def _warmup(self):
        """预热模型，加速第一次推理"""
        import torch
        from PIL import Image
        
        # 创建一个简单的测试图片
        dummy_image = Image.new('RGB', (224, 224), color='white')
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "描述这张图片"},
                    {"type": "image", "image": dummy_image}
                ]
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text],
            images=dummy_image,
            return_tensors="pt",
            padding=True
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # 执行一次快速推理
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                _ = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                )
        
        # 清理
        del inputs
        if device == "cuda":
            torch.cuda.empty_cache()

    def _detect_image_format(self, content: bytes) -> str:
        """根据图片内容检测格式"""
        # WebP: RIFF....WEBP
        if content[:4] == b'RIFF' and content[8:12] == b'WEBP':
            return '.webp'
        # JPEG: FF D8 FF
        elif content[:2] == b'\xff\xd8':
            return '.jpg'
        # PNG: 89 50 4E 47
        elif content[:4] == b'\x89PNG':
            return '.png'
        # GIF: 47 49 46 38
        elif content[:4] == b'GIF8':
            return '.gif'
        else:
            return '.jpg'

    def _download_image(self, image_path: str) -> str:
        """下载图片到本地临时目录"""
        if not image_path.startswith(('http://', 'https://')):
            return image_path
        
        print(f"正在下载图片: {image_path}")
        
        # 下载图片
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Referer': 'https://www.xiaohongshu.com/',
        }
        
        try:
            response = requests.get(image_path, headers=headers, timeout=30)
            response.raise_for_status()
        except Exception as e:
            print(f"[ERROR] 下载失败: {e}")
            raise
        
        # 检测格式
        ext = self._detect_image_format(response.content)
        
        temp_filename = f"{uuid.uuid4().hex}{ext}"
        temp_path = os.path.join(self.temp_dir, temp_filename)
        
        # 保存到本地
        with open(temp_path, 'wb') as f:
            f.write(response.content)
        
        # 验证
        try:
            img = Image.open(temp_path)
            img.verify()
            print(f"图片下载完成: {temp_path} (格式: {ext})")
        except Exception as e:
            print(f"[ERROR] 下载的文件不是有效图片: {e}")
            os.remove(temp_path)
            raise ValueError(f"下载的文件无效: {e}")
        
        return temp_path

    def _load_image(self, image_path: str) -> Image.Image:
        """加载图像"""
        if image_path.startswith(('http://', 'https://')):
            local_path = self._download_image(image_path)
            image = Image.open(local_path).convert('RGB')
        else:
            image = Image.open(image_path).convert('RGB')
        return image

    def process(self, image_path: str) -> Dict[str, any]:
        """处理图像，提取名词和形容词"""
        print(f"正在处理图像: {image_path}")

        if not self._loaded:
            self.load()

        local_image_path = None
        content = ""
        try:
            # 下载图片
            if image_path.startswith(('http://', 'https://')):
                local_image_path = self._download_image(image_path)
                process_path = local_image_path
            else:
                process_path = image_path

            if self.provider == "huggingface":
                content = self._process_huggingface(process_path)
            elif self.provider == "ollama":
                content = self._process_ollama(process_path)
            elif self.provider == "openrouter":
                content = self._process_openrouter(process_path)
            else:
                raise ValueError(f"不支持的provider: {self.provider}")

            nouns, adjectives = self._parse_response(content)

        except Exception as e:
            print(f"图像处理失败: {str(e)}")
            import traceback
            traceback.print_exc()
            nouns = []
            adjectives = []

        finally:
            # 清理临时文件
            if local_image_path and os.path.exists(local_image_path):
                try:
                    os.remove(local_image_path)
                except:
                    pass

        return {
            "nouns": nouns,
            "adjectives": adjectives,
            "noun_string": " ".join(nouns),
            "adjective_string": " ".join(adjectives),
            "raw_response": content if 'content' in dir() else ""
        }

    def _process_huggingface(self, image_path: str) -> str:
        """使用本地HuggingFace模型处理图像"""
        import torch
        import time
        
        print(f"[DEBUG] 处理图像: {image_path}")
        t0 = time.time()
        
        # 加载图片
        image = Image.open(image_path).convert('RGB')
        print(f"[DEBUG] 图片加载完成, 尺寸: {image.size}, 耗时: {time.time()-t0:.2f}s")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.DEFAULT_PROMPT},
                    {"type": "image", "image": image}
                ]
            }
        ]

        # 处理输入
        t1 = time.time()
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        print(f"[DEBUG] 输入处理完成, 耗时: {time.time()-t1:.2f}s")

        # 生成
        print(f"[DEBUG] 开始生成 (max_new_tokens=256)...")
        if device == "cuda":
            print(f"[DEBUG] 生成前显存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        t2 = time.time()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )
        
        # 清理
        del inputs
        if device == "cuda":
            torch.cuda.empty_cache()
        
        print(f"[DEBUG] 生成完成, 耗时: {time.time()-t2:.2f}s")

        # 解码
        input_ids = inputs.get("input_ids") if 'inputs' in dir() else None
        if input_ids is not None:
            output_ids = output_ids[:, input_ids.size(1):]
        
        content = self.processor.batch_decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]
        
        print(f"[DEBUG] 总耗时: {time.time()-t0:.2f}s")
        return content

    def _process_ollama(self, image_path: str) -> str:
        """使用Ollama处理图像"""
        import base64
        
        # 读取并编码图片
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": self.DEFAULT_PROMPT,
            "images": [image_data],
            "stream": False
        }
        
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        
        return response.json().get("response", "")

    def _process_openrouter(self, image_path: str) -> str:
        """使用OpenRouter API处理图像"""
        import base64
        
        # 读取并编码图片
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "qwen/qwen2-vl-7b-instruct",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.DEFAULT_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                    ]
                }
            ],
            "max_tokens": 512
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        
        return response.json()["choices"][0]["message"]["content"]

    def _parse_response(self, content: str) -> tuple:
        """解析模型输出，提取名词和形容词"""
        import re
        
        # 尝试提取JSON
        nouns = []
        adjectives = []
        
        # 方法1: 直接解析
        try:
            # 找到JSON块
            json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                nouns = data.get("nouns", [])
                adjectives = data.get("adjectives", [])
                return nouns, adjectives
        except:
            pass
        
        # 方法2: 尝试修复不完整的JSON
        try:
            # 尝试找到完整的 JSON
            if '{' in content and '}' not in content:
                content = content + '}'
            if '}' in content and '{' not in content:
                content = '{' + content
            
            # 查找所有 JSON 对象
            json_matches = re.findall(r'\{[^{}]*\}', content, re.DOTALL)
            for match in json_matches:
                try:
                    data = json.loads(match)
                    if "nouns" in data and "adjectives" in data:
                        nouns = data["nouns"]
                        adjectives = data["adjectives"]
                        return nouns, adjectives
                except:
                    continue
        except:
            pass
        
        # 方法3: 使用正则提取
        try:
            nouns_match = re.search(r'"nouns"\s*:\s*\[([^\]]*)\]', content)
            if nouns_match:
                nouns_str = nouns_match.group(1)
                nouns = re.findall(r'"([^"]+)"', nouns_str)
            
            adj_match = re.search(r'"adjectives"\s*:\s*\[([^\]]*)\]', content)
            if adj_match:
                adj_str = adj_match.group(1)
                adjectives = re.findall(r'"([^"]+)"', adj_str)
        except:
            pass
        
        # 如果都失败，返回空列表
        if not nouns and not adjectives:
            print(f"[WARN] 无法解析模型输出: {content[:200]}...")
        
        return nouns, adjectives
