"""
文本处理模块
使用HanLP进行词法分析，从文本中抽取名词和形容词
"""

import hanlp
import os
from typing import List, Dict, Tuple


class TextProcessor:
    """文本处理器 - 使用HanLP抽取名词和形容词"""

    def __init__(self, pretrained: str = "standard", cache_dir: str = None):
        """
        初始化文本处理器

        Args:
            pretrained: HanLP预训练模型名称
            cache_dir: 模型缓存目录（默认项目目录的models/hanlp）
        """
        self.pretrained = pretrained

        # 设置模型缓存目录
        if cache_dir:
            self.cache_dir = cache_dir
        else:
            self.cache_dir = "./models/hanlp"

        # 创建缓存目录
        os.makedirs(self.cache_dir, exist_ok=True)

        # 设置HanLP缓存目录
        os.environ['HANLP_HOME'] = self.cache_dir

        self.tokenizer = None
        self.pos_tagger = None
        self._loaded = False

    def load(self):
        """加载HanLP模型（延迟加载，首次使用时加载）"""
        if self._loaded:
            return

        print(f"正在加载HanLP模型: {self.pretrained}")
        print(f"模型缓存目录: {self.cache_dir}")

        # 加载中文分词模型
        self.tokenizer = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)

        # 加载词性标注模型
        self.pos_tagger = hanlp.load(hanlp.pretrained.pos.CTB5_POS_RADICAL_ELECTRA_SMALL)

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

        # 分词
        tokens = self.tokenizer(text)

        # 词性标注
        pos_tags = self.pos_tagger(tokens)

        # 抽取名词和形容词
        nouns = []
        adjectives = []

        for token, pos in zip(tokens, pos_tags):
            # 过滤单字
            if len(token) < 2:
                continue

            # 名词类识别
            if self._is_noun(pos):
                nouns.append(token)
            # 形容词类识别
            elif self._is_adjective(pos):
                adjectives.append(token)

        return nouns, adjectives

    def _is_noun(self, pos: str) -> bool:
        """判断是否为名词"""
        noun_tags = ['NN', 'NR', 'NS', 'NT', 'NZ', 'NI', 'NL', 'NRG', 'NSG', 'NTG']
        return pos in noun_tags or pos.startswith('N')

    def _is_adjective(self, pos: str) -> bool:
        """判断是否为形容词"""
        adj_tags = ['VA', 'VAC', 'JJ', 'VA', 'VAC']
        return pos in adj_tags or pos.startswith('VA') or pos == 'JJ'

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
