"""
文本处理模块
使用HanLP进行词法分析，从文本中抽取名词和形容词
"""

import hanlp
import os
from typing import List, Dict, Tuple


class TextProcessor:
    """文本处理器 - 使用HanLP抽取名词和形容词"""

    def __init__(self, cache_dir: str = None):
        """
        初始化文本处理器

        Args:
            cache_dir: 模型缓存目录（默认项目目录的models/hanlp）
        """
        # 设置模型缓存目录
        if cache_dir:
            self.cache_dir = cache_dir
        else:
            self.cache_dir = "./models/hanlp"

        # 创建缓存目录
        os.makedirs(self.cache_dir, exist_ok=True)

        self.pipeline = None
        self._loaded = False

    def load(self):
        """加载HanLP模型（延迟加载，首次使用时加载）"""
        if self._loaded:
            return

        print(f"正在加载HanLP模型...")
        print(f"模型缓存目录: {self.cache_dir}")

        # 使用 HanLP 2.0 的 Pipeline API
        # 这是一个轻量级的多任务模型，可以同时进行分词和词性标注
        try:
            # 尝试使用 MTB（多任务模型）方式
            self.pipeline = hanlp.load(hanlp.pretrained.mtl.ko_electra_base_grit)
        except:
            try:
                # 备选方案：使用中文ELECTRA模型
                self.pipeline = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
            except Exception as e:
                print(f"模型加载警告: {e}")
                print("尝试使用基础模型...")
                # 最后备选方案
                self.pipeline = hanlp.load(hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH)

        self._loaded = True
        print("HanLP模型加载完成")

    def extract_words(self, text: str) -> Tuple[List[str], List[str]]:
        """
        从文本中抽取名词和形容词

        Args:
            text: 输入文本

        Returns:
            (名词列表, 形容词列表)
        """
        if not self._loaded:
            self.load()

        # 分词 + 词性标注
        try:
            # 尝试使用 pipeline 方式
            if hasattr(self.pipeline, 'predict'):
                # 新版 HanLP API
                result = self.pipeline(text)
            else:
                # 旧版 API
                result = self.pipeline(text)
        except Exception as e:
            print(f"分词失败: {e}")
            return [], []

        # 解析结果
        nouns = []
        adjectives = []

        # 兼容不同的输出格式
        if isinstance(result, dict):
            # 字典格式
            if 'tok/fine' in result:
                tokens = result['tok/fine']
            elif 'tok/coarse' in result:
                tokens = result['tok/coarse']
            else:
                tokens = list(result.values())[0] if result else []

            if 'pos/ctb' in result:
                pos_tags = result['pos/ctb']
            else:
                pos_tags = list(result.values())[1] if len(result) > 1 else []
        elif hasattr(result, 'tolist'):
            # DataFrame 格式
            tokens = result['token'].tolist()
            pos_tags = result['pos'].tolist()
        elif isinstance(result, list):
            # 列表格式 - 假设是 (token, pos) 元组列表
            tokens = [r[0] for r in result]
            pos_tags = [r[1] for r in result]
        else:
            # 尝试其他方式
            tokens = text  # 退化为单字
            pos_tags = []

        # 如果 tokens 是字符串（未分词），则按字符分割
        if isinstance(tokens, str):
            tokens = list(tokens)
            pos_tags = []

        for i, token in enumerate(tokens):
            # 过滤单字
            if len(token) < 2:
                continue

            # 获取词性
            pos = pos_tags[i] if i < len(pos_tags) else ''

            # 名词类识别
            if self._is_noun(pos):
                nouns.append(token)
            # 形容词类识别
            elif self._is_adjective(pos):
                adjectives.append(token)

        return nouns, adjectives

    def _is_noun(self, pos: str) -> bool:
        """判断是否为名词"""
        if not pos:
            return False
        noun_tags = ['NN', 'NR', 'NS', 'NT', 'NZ', 'NI', 'NL', 'NRG', 'NSG', 'NTG', 'n', 'nr', 'ns', 'nt', 'nz']
        return pos in noun_tags or pos.startswith('N') or pos in noun_tags

    def _is_adjective(self, pos: str) -> bool:
        """判断是否为形容词"""
        if not pos:
            return False
        adj_tags = ['VA', 'VAC', 'JJ', 'VA', 'VAC', 'a', 'ad', 'an']
        return pos in adj_tags or pos.startswith('VA') or pos == 'JJ' or pos.startswith('A')

    def process(self, text: str) -> Dict[str, any]:
        """
        处理文本，返回结构化结果

        Args:
            text: 输入文本

        Returns:
            {
                "nouns": [...],
                "adjectives": [...],
                "noun_string": "...",
                "adjective_string": "..."
            }
        """
        nouns, adjectives = self.extract_words(text)

        return {
            "nouns": nouns,
            "adjectives": adjectives,
            "noun_string": " ".join(nouns),
            "adjective_string": " ".join(adjectives)
        }


def test():
    """测试函数"""
    processor = TextProcessor()
    processor.load()

    # 测试文本
    test_texts = [
        "这家酒店位于繁华的市中心，装修豪华，服务周到",
        "风景优美的古镇，蓝天白云，古色古香的建筑",
        "海鲜新鲜美味，价格实惠，环境优雅"
    ]

    print("=" * 50)
    print("文本处理模块测试")
    print("=" * 50)

    for text in test_texts:
        result = processor.process(text)
        print(f"\n原文: {text}")
        print(f"名词: {result['nouns']}")
        print(f"形容词: {result['adjectives']}")


if __name__ == "__main__":
    test()
