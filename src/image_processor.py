"""
图像处理模块
使用Qwen2.5-VL生成图像的关键词项（名词+形容词）
支持Ollama本地运行和OpenRouter云端API
"""

import base64
import json
import os
import requests
from typing import Dict, List, Optional
from PIL import Image
import io


class ImageProcessor:
    """图像处理器 - 提取图像中的名词和形容词"""

    # 默认prompt，指导模型输出格式
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
        provider: str = "ollama",
        model: str = "qwen2.5-vl:7b-instruct",
        base_url: str = "http://localhost:11434",
        api_key: Optional[str] = None
    ):
        """
        初始化图像处理器

        Args:
            provider: "ollama"（本地）或 "openrouter"（云端API）
            model: 模型名称
            base_url: API地址
            api_key: API密钥
        """
        self.provider = provider
        self.model = model
        self.base_url = base_url

        # 优先使用环境变量中的API Key
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")

        if not self.api_key and provider == "openrouter":
            print("警告: 未设置OPENROUTER_API_KEY环境变量，云端API可能无法使用")

    def load_image(self, image_path: str) -> str:
        """
        加载图像并转换为base64

        Args:
            image_path: 图像文件路径或URL

        Returns:
            base64编码的图像字符串
        """
        # 支持URL和本地路径
        if image_path.startswith('http://') or image_path.startswith('https://'):
            # 下载网络图片
            response = requests.get(image_path, timeout=30)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content))
        else:
            # 读取本地图片
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"图片文件不存在: {image_path}")
            image = Image.open(image_path)

        # 转换为base64
        buffered = io.BytesIO()
        # 统一转为RGB模式
        if image.mode in ('RGBA', 'P'):
            image = image.convert('RGB')
        image.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return img_str

    def _call_api(self, image_path: str) -> str:
        """调用视觉模型API"""
        img_str = self.load_image(image_path)

        # 构建消息
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

        if self.provider == "ollama":
            return self._call_ollama(messages)
        elif self.provider == "openrouter":
            return self._call_openrouter(messages)
        else:
            raise ValueError(f"不支持的provider: {self.provider}")

    def _call_ollama(self, messages: List) -> str:
        """调用本地Ollama API"""
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False
            },
            timeout=120
        )

        if response.status_code != 200:
            raise Exception(f"Ollama API错误: {response.status_code} - {response.text}")

        result = response.json()
        return result["message"]["content"]

    def _call_openrouter(self, messages: List) -> str:
        """调用OpenRouter API（云端）"""
        if not self.api_key:
            raise ValueError("OpenRouter需要设置API Key")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com",  # OpenRouter需要
            "X-Title": "ImageTextSimilarity"
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json={
                "model": self.model,
                "messages": messages
            },
            timeout=120
        )

        if response.status_code != 200:
            raise Exception(f"OpenRouter API错误: {response.status_code} - {response.text}")

        result = response.json()
        return result["choices"][0]["message"]["content"]

    def process(self, image_path: str) -> Dict[str, any]:
        """
        处理图像，提取名词和形容词

        Args:
            image_path: 图像路径（本地路径或URL）

        Returns:
            {
                "nouns": [...],
                "adjectives": [...],
                "noun_string": "...",
                "adjective_string": "..."
            }
        """
        print(f"正在处理图像: {image_path}")

        try:
            # 调用API
            content = self._call_api(image_path)

            # 解析JSON响应
            nouns, adjectives = self._parse_response(content)

        except Exception as e:
            print(f"图像处理失败: {str(e)}")
            nouns = []
            adjectives = []

        return {
            "nouns": nouns,
            "adjectives": adjectives,
            "noun_string": " ".join(nouns),
            "adjective_string": " ".join(adjectives),
            "raw_response": content if 'content' in dir() else ""
        }

    def _parse_response(self, content: str) -> Tuple[List[str], List[str]]:
        """解析API响应，提取名词和形容词列表"""
        try:
            # 清理响应内容，提取JSON
            content = content.strip()

            # 处理可能的markdown代码块
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            # 尝试提取JSON对象
            # 找到第一个{和最后一个}
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
    # 测试配置 - 使用OpenRouter
    processor = ImageProcessor(
        provider="openrouter",
        model="qwen/qwen2.5-vl-7b-instruct"
    )

    # 检查API Key
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if api_key:
        print(f"API Key已设置: {api_key[:10]}...")
    else:
        print("警告: 请设置OPENROUTER_API_KEY环境变量")

    # 测试解析功能
    test_json = '{"nouns": ["天空", "城市", "建筑"], "adjectives": ["美丽", "繁华"]}'
    nouns, adjectives = processor._parse_response(test_json)
    print(f"\n解析测试: 名词={nouns}, 形容词={adjectives}")


if __name__ == "__main__":
    test()
