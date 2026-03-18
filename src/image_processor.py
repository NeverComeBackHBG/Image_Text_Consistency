"""
图像处理模块
使用Qwen2.5-VL生成图像的关键词项（名词+形容词）
支持四种方式：AutoDL（推荐）、Ollama、HuggingFace、OpenRouter API
"""

import json
import os
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
        local_model_path: str = "/root/models/Qwen2.5-VL-7B-Instruct"
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
        """
        self.provider = provider
        self.model_name = model
        self.base_url = base_url
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.cache_dir = cache_dir
        self.local_model_path = local_model_path
        self._loaded = False

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

    def _load_image_from_url(self, url: str) -> Image.Image:
        """从URL加载图像"""
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        # 关键修复：使用 io.BytesIO 包装二进制数据
        image = Image.open(io.BytesIO(response.content)).convert('RGB')
        return image

    def _load_image_from_path(self, path: str) -> Image.Image:
        """从本地路径加载图像"""
        image = Image.open(path).convert('RGB')
        return image

    def process(self, image_path: str) -> Dict[str, any]:
        """处理图像，提取名词和形容词"""
        print(f"正在处理图像: {image_path}")

        if not self._loaded:
            self.load()

        try:
            if self.provider == "autodl":
                content = self._process_autodl(image_path)
            elif self.provider == "ollama":
                content = self._process_ollama(image_path)
            elif self.provider == "huggingface":
                content = self._process_huggingface(image_path)
            elif self.provider == "openrouter":
                content = self._process_openrouter(image_path)
            else:
                raise ValueError(f"不支持的provider: {self.provider}")

            nouns, adjectives = self._parse_response(content)

        except Exception as e:
            print(f"图像处理失败: {str(e)}")
            import traceback
            traceback.print_exc()
            nouns = []
            adjectives = []

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

        # 加载图像
        if image_path.startswith('http://') or image_path.startswith('https://'):
            image = self._load_image_from_url(image_path)
        else:
            image = self._load_image_from_path(image_path)

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

        # 去除输入部分
        input_ids = inputs.get("input_ids")
        if input_ids is not None:
            output_ids = output_ids[:, input_ids.size(1):]

        output_text = self.processor.batch_decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]

        return output_text

    def _process_ollama(self, image_path: str) -> str:
        """使用本地Ollama处理图像"""
        # 加载图像为base64
        if image_path.startswith('http://') or image_path.startswith('https://'):
            image = self._load_image_from_url(image_path)
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')
            img_bytes = img_byte_arr.getvalue()
        else:
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
        if image_path.startswith('http://') or image_path.startswith('https://'):
            image = self._load_image_from_url(image_path)
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')
            img_bytes = img_byte_arr.getvalue()
        else:
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
        """解析响应，提取名词和形容词"""
        try:
            content = content.strip()

            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            start = content.find('{')
            end = content.rfind('}')
            if start != -1 and end != -1:
                content = content[start:end+1]

            result = json.loads(content.strip())

            nouns = result.get("nouns", [])
            adjectives = result.get("adjectives", [])

            return nouns, adjectives

        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")
            print(f"原始响应（前200字符）: {content[:200]}")
            return [], []


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
