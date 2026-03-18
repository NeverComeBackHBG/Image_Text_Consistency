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
        print("视觉模型就绪")

    def _load_autodl_model(self):
        """从AutoDL社区镜像加载本地Qwen2.5-VL模型"""
        import torch
        from transformers import AutoProcessor, AutoModelForVision2Seq

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"正在加载 AutoDL 本地模型: {self.local_model_path}")
        print(f"使用设备: {device}")

        self.processor = AutoProcessor.from_pretrained(
            self.local_model_path,
            trust_remote_code=True
        )

        self.model = AutoModelForVision2Seq.from_pretrained(
            self.local_model_path,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else "cpu",
            trust_remote_code=True
        )

        if device == "cpu":
            self.model = self.model.to("cpu")

        print("AutoDL Qwen2.5-VL 模型加载完成")

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
        
        # 下载图片
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(image_path, headers=headers, timeout=30)
        response.raise_for_status()
        
        # 根据实际内容检测格式
        ext = self._detect_image_format(response.content)
        
        temp_filename = f"{uuid.uuid4().hex}{ext}"
        temp_path = os.path.join(self.temp_dir, temp_filename)
        
        # 保存到本地
        with open(temp_path, 'wb') as f:
            f.write(response.content)
        
        print(f"图片下载完成: {temp_path} (格式: {ext})")
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

        # 直接加载本地图片
        image = Image.open(image_path).convert('RGB')

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.DEFAULT_PROMPT},
                    {"type": "image", "image": image}
                ]
            }
        ]

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

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

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
