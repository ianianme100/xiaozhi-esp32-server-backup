"""輕量級本地語意相似度工具。

專案目前沒有安裝本地 embedding 模型（sentence-transformers 等），呼叫遠端 embedding
API 又會替每一句話增加延遲與外部依賴。這裡用 character bigram 向量 + cosine
similarity 做一個不需要額外套件、可離線運作的近似語意比對：比對的是「這句話跟範例
句子像不像」，而不是「這句話裡有沒有出現某個關鍵字」，因此可以承受語序、口語化、
同義詞變化（例如「房間有點悶」「空氣不太好」都能比對到同一類範例）。

如果未來要接上真正的 embedding 模型或 API，只要保留 `best_match` 的介面
（輸入文字 + 各意圖的範例句字典，輸出最相似的標籤與分數），即可原地替換實作。
"""

from collections import Counter
from typing import Dict, List, Optional, Tuple


def _bigram_vector(text: str) -> Counter:
    normalized = text.lower().replace(" ", "")
    if len(normalized) < 2:
        return Counter([normalized]) if normalized else Counter()
    return Counter(normalized[i : i + 2] for i in range(len(normalized) - 1))


def cosine_similarity(text_a: str, text_b: str) -> float:
    vec_a = _bigram_vector(text_a)
    vec_b = _bigram_vector(text_b)
    if not vec_a or not vec_b:
        return 0.0

    common = set(vec_a) & set(vec_b)
    dot = sum(vec_a[k] * vec_b[k] for k in common)
    if dot == 0:
        return 0.0

    norm_a = sum(v * v for v in vec_a.values()) ** 0.5
    norm_b = sum(v * v for v in vec_b.values()) ** 0.5
    return dot / (norm_a * norm_b)


def best_match(
    text: str, examples: Dict[str, List[str]]
) -> Tuple[Optional[str], float]:
    """回傳跟 text 語意最相近的標籤與相似度分數。

    examples: { 標籤: [範例句, ...] }
    比對方式：取 text 與每個標籤底下所有範例句相似度的最大值，分數最高的標籤勝出。
    """
    best_label = None
    best_score = 0.0
    for label, sentences in examples.items():
        score = max((cosine_similarity(text, s) for s in sentences), default=0.0)
        if score > best_score:
            best_label = label
            best_score = score
    return best_label, best_score
