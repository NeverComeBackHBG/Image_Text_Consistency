"""
图像处理模块
使用Qwen2.5-VL生成图像的关键词项（名词+形容词）
支持四种方式：AutoDL（推荐）、Ollama、HuggingFace、OpenRouter API
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

    def __init__(
        self,
        provider: str = "autodl",
        model: str = "qwen2.5-vl:7b-instruct",
        base_url: str = "http://localhost:11434",
        api_key: Optional[str] = None,
        cache_dir: str = "./models",
        local_model_path: str = "/root/models/Qwen2.5-VL-7B-Instruct",
        temp_dir: str = "./temp_images"
    ):
        """
        初始化图像处理器

        Args:
            provider: "autodl"（推荐）、"ollama"、"huggingface"、"openrouter"
            model: 模型名称
            base_url: API地址（Ollama/OpenRouter用）
            api_key: API密钥
            cache_dir: 模型缓存目录
            local_model_path: AutoDL本地模型路径
            temp_dir: 临时图片下载目录
        """
        self.provider = provider
        self.model_name = model
        self.base_url = base_url
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.cache_dir = cache_dir
        self.local_model_path = local_model_path
        self.temp_dir = temp_dir
        self._loaded = False

        # 创建临时目录
        os.makedirs(self.temp_dir, exist_ok=True)

    def load(self):
        """加载模型"""
        if self._loaded:
            return

        print(f"视觉模型配置: {self.provider} - {self.model_name}")

        if self.provider == "autodl":
            self._load_autodl_model()
        elif self.provider == "huggingface":
            self._load_huggingface_model()

        self._loaded = True
        
        # 预热模型（可选，加速第一次推理）
        print("模型预热中...")
        try:
            self._warmup()
            print("模型预热完成")
        except Exception as e:
            print(f"预热失败: {e}")
        
        print("视觉模型就绪")

    def _load_autodl_model(self):
        """从AutoDL社区镜像加载本地Qwen2.5-VL模型"""
        import torch
        from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"正在加载 AutoDL 本地模型: {self.local_model_path}")
        print(f"使用设备: {device}")
        
        # 先清理显存
        if device == "cuda":
            torch.cuda.empty_cache()
            print(f"当前显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

        self.processor = AutoProcessor.from_pretrained(
            self.local_model_path,
            trust_remote_code=True
        )

        if device == "cuda":
            # 先尝试正常加载（你的 24GB 显存应该够）
            try:
                print("尝试使用 bfloat16 加载...")
                torch.cuda.empty_cache()
                self.model = AutoModelForVision2Seq.from_pretrained(
                    self.local_model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True
                )
                print("bfloat16 加载成功!")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("显存不足，尝试 8-bit 量化...")
                    torch.cuda.empty_cache()
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0,
                    )
                    self.model = AutoModelForVision2Seq.from_pretrained(
                        self.local_model_path,
                        quantization_config=quantization_config,
                        device_map="auto",
                        trust_remote_code=True
                    )
                    print("8-bit 量化加载成功!")
                else:
                    raise
        else:
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.local_model_path,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True
            )

        print("AutoDL Qwen2.5-VL 模型加载完成")

    def _warmup(self):
        """预热模型，加速第一次推理"""
        import torch
        from PIL import Image
        import io
        
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
            _ = self.model.generate(
                **inputs,
                max_new_tokens=10,  # 只生成少量 token 预热
                do_sample=False,
            )

    def _load_huggingface_model(self):
        """从HuggingFace加载模型"""
        from transformers import AutoProcessor, AutoModelForVision2Seq
        import torch

        os.makedirs(self.cache_dir, exist_ok=True)

        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir
        )

        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=self.cache_dir
        )

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
            return '.jpg'  # 默认
    
    def _download_image(self, image_path: str) -> str:
        """
        下载图片到本地临时目录
        
        Returns:
            本地图片路径
        """
        # 如果是本地路径，直接返回
        if not image_path.startswith(('http://', 'https://')):
            return image_path
        
        print(f"正在下载图片: {image_path}")
        
        # 下载图片 - 添加更多 headers 避免防盗链
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
        
        # 检查内容类型
        content_type = response.headers.get('Content-Type', '')
        print(f"[DEBUG] Content-Type: {content_type}")
        
        # 检查是否是图片
        if not content_type.startswith('image/'):
            # 可能是 HTML 错误页面
            preview = response.content[:200]
            print(f"[ERROR] 返回的不是图片! 内容预览: {preview}")
            raise ValueError(f"下载的不是图片，Content-Type: {content_type}")
        
        # 根据实际内容检测格式
        ext = self._detect_image_format(response.content)
        
        temp_filename = f"{uuid.uuid4().hex}{ext}"
        temp_path = os.path.join(self.temp_dir, temp_filename)
        
        # 保存到本地
        with open(temp_path, 'wb') as f:
            f.write(response.content)
        
        # 验证文件是否有效
        try:
            from PIL import Image
            img = Image.open(temp_path)
            img.verify()  # 验证图片完整性
            print(f"图片下载完成: {temp_path} (格式: {ext}, 尺寸: {img.size})")
        except Exception as e:
            print(f"[ERROR] 下载的文件不是有效图片: {e}")
            os.remove(temp_path)
            raise ValueError(f"下载的文件无效: {e}")
        
        return temp_path

    def _load_image(self, image_path: str) -> Image.Image:
        """加载图像（支持URL和本地路径）"""
        if image_path.startswith(('http://', 'https://')):
            # 先下载到本地
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
        try:
            # 如果是URL，先下载
            if image_path.startswith(('http://', 'https://')):
                local_image_path = self._download_image(image_path)
                process_path = local_image_path
            else:
                process_path = image_path

            if self.provider == "autodl":
                content = self._process_autodl(process_path)
            elif self.provider == "ollama":
                content = self._process_ollama(process_path)
            elif self.provider == "huggingface":
                content = self._process_huggingface(process_path)
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
                    print(f"已清理临时文件: {local_image_path}")
                except:
                    pass

        return {
            "nouns": nouns,
            "adjectives": adjectives,
            "noun_string": " ".join(nouns),
            "adjective_string": " ".join(adjectives),
            "raw_response": content if 'content' in dir() else ""
        }

    def _process_autodl(self, image_path: str) -> str:
        """使用AutoDL本地Qwen2.5-VL模型处理图像"""
        import torch
        import time
        
        print(f"[DEBUG] _process_autodl 接收路径: {image_path}")
        
        # 直接加载本地图片
        print(f"[DEBUG] 开始加载图片...")
        t0 = time.time()
        image = Image.open(image_path).convert('RGB')
        print(f"[DEBUG] 图片加载完成, 耗时: {time.time()-t0:.2f}s, 尺寸: {image.size}")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.DEFAULT_PROMPT},
                    {"type": "image", "image": image}
                ]
            }
        ]

        print(f"[DEBUG] 开始 apply_chat_template...")
        t1 = time.time()
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        print(f"[DEBUG] chat_template 完成, 耗时: {time.time()-t1:.2f}s")

        print(f"[DEBUG] 开始处理输入...")
        t2 = time.time()
        inputs = self.processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding=True
        )
        print(f"[DEBUG] 输入处理完成, 耗时: {time.time()-t2:.2f}s")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[DEBUG] 使用设备: {device}")
        if device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        print(f"[DEBUG] 开始模型生成 (max_new_tokens=256)...")
        print(f"[DEBUG] 生成前显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        t3 = time.time()
        
        # 使用 torch.cuda.amp.autocast() 减少显存使用
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )
        
        # 立即释放输入张量
        del inputs
        torch.cuda.empty_cache()
        
        print(f"[DEBUG] 模型生成完成, 耗时: {time.time()-t3:.2f}s")
        print(f"[DEBUG] 生成后显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

        input_ids = inputs.get("input_ids")
        if input_ids is not None:
            output_ids = output_ids[:, input_ids.size(1):]

        output_text = self.processor.batch_decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]

        return output_text

    def _process_ollama(self, image_path: str) -> str:
        """使用本地Ollama处理图像"""
        with open(image_path, 'rb') as f:
            img_bytes = f.read()

        img_str = base64.b64encode(img_bytes).decode()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.DEFAULT_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_str}"
                        }
                    }
                ]
            }
        ]

        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model_name,
                "messages": messages,
                "stream": False
            },
            timeout=120
        )

        if response.status_code != 200:
            raise Exception(f"Ollama API错误: {response.status_code} - {response.text}")

        result = response.json()
        return result["message"]["content"]

    def _process_huggingface(self, image_path: str) -> str:
        """使用HuggingFace模型处理图像"""
        from qwen_vl_utils import process_vision_info
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": self.DEFAULT_PROMPT}
                ]
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False
            )

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs['input_ids'], output_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]

        return output_text

    def _process_openrouter(self, image_path: str) -> str:
        """使用OpenRouter API处理图像"""
        with open(image_path, 'rb') as f:
            img_bytes = f.read()

        img_str = base64.b64encode(img_bytes).decode()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.DEFAULT_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_str}"
                        }
                    }
                ]
            }
        ]

        if not self.api_key:
            raise ValueError("OpenRouter需要设置API Key")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com",
            "X-Title": "ImageTextSimilarity"
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json={
                "model": self.model_name,
                "messages": messages
            },
            timeout=120
        )

        if response.status_code != 200:
            raise Exception(f"OpenRouter API错误: {response.status_code} - {response.text}")

        result = response.json()
        return result["choices"][0]["message"]["content"]

    def _parse_response(self, content: str) -> tuple:
        """解析响应，提取名词和形容词（增强容错）"""
        try:
            content = content.strip()

            # 清理 markdown 格式
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            # 提取 JSON 部分
            start = content.find('{')
            end = content.rfind('}')
            if start != -1 and end != -1:
                content = content[start:end+1]

            # 尝试解析 JSON
            try:
                result = json.loads(content.strip())
            except json.JSONDecodeError as e:
                # JSON 不完整，尝试修复
                print(f"JSON不完整，尝试修复...")
                content = self._fix_incomplete_json(content)
                result = json.loads(content)

            nouns = result.get("nouns", [])
            adjectives = result.get("adjectives", [])

            return nouns, adjectives

        except Exception as e:
            print(f"JSON解析失败: {e}")
            print(f"原始响应: {content[:500]}")
            # 尝试正则提取
            return self._extract_with_regex(content)

    def _fix_incomplete_json(self, content: str) -> str:
        """修复不完整的 JSON"""
        content = content.strip()

        # 找到最后一个完整的元素
        last_bracket = content.rfind(']')
        last_brace = content.rfind('}')

        if last_bracket > last_brace:
            # 数组先结束
            content = content[:last_bracket+1]
            # 补全对象
            if content.rfind('"adjectives"') > content.rfind('"nouns"'):
                content = content + "}"
            else:
                content = content + ', "adjectives": []}'
        else:
            # 对象结束
            content = content[:last_brace+1]

        # 确保有开头的 {
        if not content.startswith('{'):
            content = '{' + content

        # 确保有结尾的 }
        if not content.endswith('}'):
            content = content + '}'

        return content

    def _extract_with_regex(self, content: str) -> tuple:
        """使用正则表达式提取名词和形容词"""
        nouns = []
        adjectives = []

        try:
            # 提取 nouns 数组
            nouns_match = re.search(r'"nouns"\s*:\s*\[(.*?)\]', content, re.DOTALL)
            if nouns_match:
                nouns_str = nouns_match.group(1)
                # 提取引号中的字符串
                nouns = re.findall(r'"([^"]*)"', nouns_str)

            # 提取 adjectives 数组
            adjectives_match = re.search(r'"adjectives"\s*:\s*\[(.*?)\]', content, re.DOTALL)
            if adjectives_match:
                adjectives_str = adjectives_match.group(1)
                adjectives = re.findall(r'"([^"]*)"', adjectives_str)

            print(f"使用正则提取成功: {len(nouns)} 个名词, {len(adjectives)} 个形容词")

        except Exception as e:
            print(f"正则提取也失败: {e}")

        return nouns, adjectives

    def cleanup_temp(self):
        """清理所有临时文件"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            os.makedirs(self.temp_dir, exist_ok=True)
            print(f"已清理临时目录: {self.temp_dir}")


def test():
    """测试函数"""
    processor = ImageProcessor(
        provider="autodl",
        local_model_path="/root/models/Qwen2.5-VL-7B-Instruct"
    )
    print("图像处理器初始化完成")
    print(f"将使用 AutoDL 本地模型: {processor.local_model_path}")


if __name__ == "__main__":
    test()
