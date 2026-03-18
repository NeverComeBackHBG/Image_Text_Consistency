import hanlp
import os
from typing import List, Dict, Tuple

class TextProcessor:
    """文本处理器 - 使用HanLP抽取名词和形容词（2026年兼容版）"""

    def __init__(self, cache_dir: str = None):
        if cache_dir:
            self.cache_dir = cache_dir
        else:
            self.cache_dir = "./models/hanlp"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.pipeline = None
        self._loaded = False

    def load(self):
        """加载HanLP模型"""
        if self._loaded:
            return
            
        print("正在加载 HanLP 模型...")
        print(f"模型缓存目录: {self.cache_dir}")

        try:
            # 推荐使用目前最稳定的中文多任务模型
            self.pipeline = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH)
            print("✓ 成功加载 CLOSE 多任务模型")
        except:
            try:
                # 备选：分词 + 词性标注组合模型
                self.pipeline = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_ELECTRA_SMALL_ZH)
                print("✓ 成功加载 TOK+POS 小模型")
            except Exception as e:
                print(f"模型加载失败: {e}")
                # 最终保底方案
                self.pipeline = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
                print("✓ 使用基础分词模型（无词性标注）")

        self._loaded = True
        print("HanLP 模型加载完成\n")

    def extract_words(self, text: str) -> Tuple[List[str], List[str]]:
        """提取名词和形容词（增强兼容版）"""
        if not self._loaded:
            self.load()

        try:
            result = self.pipeline(text)

            nouns = []
            adjectives = []

            # ==================== 兼容多种返回格式 ====================
            if isinstance(result, dict):
                # 新版 HanLP 多任务模型常用 key
                tokens = result.get('tok/fine') or result.get('tok/coarse') or result.get('tok') or []
                pos_tags = result.get('pos/ctb') or result.get('pos/ctb9') or result.get('pos') or []

            elif isinstance(result, list):
                # 列表格式
                if result and isinstance(result[0], (list, tuple)) and len(result[0]) >= 2:
                    # [(token, pos), ...]
                    tokens = [item[0] for item in result]
                    pos_tags = [item[1] for item in result]
                else:
                    # 只有词列表的情况
                    tokens = result
                    pos_tags = [''] * len(tokens)
            else:
                tokens = list(text)
                pos_tags = [''] * len(tokens)

            # ==================== 词性分类 ====================
            for token, pos in zip(tokens, pos_tags):
                if len(token.strip()) < 2:
                    continue
                    
                pos = str(pos).upper()

                if self._is_noun(pos):
                    nouns.append(token)
                elif self._is_adjective(pos):
                    adjectives.append(token)

            return nouns, adjectives

        except Exception as e:
            print(f"提取词语时发生错误: {e}")
            # 保底方案：简单分词
            tokens = [w for w in text.split() if len(w) >= 2]
            return tokens, []

    def _is_noun(self, pos: str) -> bool:
        noun_tags = {'NN', 'NR', 'NS', 'NT', 'NZ', 'N', 'NR', 'NS', 'NT'}
        return pos in noun_tags or pos.startswith('N')

    def _is_adjective(self, pos: str) -> bool:
        adj_tags = {'JJ', 'VA', 'AD', 'A', 'AN', 'VA', 'VAC'}
        return pos in adj_tags or pos.startswith('VA') or pos.startswith('A') or pos.startswith('JJ')

    def process(self, text: str) -> Dict[str, any]:
        nouns, adjectives = self.extract_words(text)
        return {
            "nouns": nouns,
            "adjectives": adjectives,
            "noun_string": " ".join(nouns),
            "adjective_string": " ".join(adjectives)
        }