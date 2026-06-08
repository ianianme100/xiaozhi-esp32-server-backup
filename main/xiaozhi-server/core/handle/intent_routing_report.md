# 語意意圖路由重構報告

## 1. 背景與動機

原本的 `core/handle/intentHandler.py` 對於燈光、紅外線、感測器、緊急呼叫四類指令，
是各自獨立的「硬解析」函式（`parse_hard_xxx` + `handle_hard_xxx`），存在兩個問題：

1. **大量重複的樣板程式碼**：每個 `handle_hard_xxx` 都重複寫了
   送出 STT 訊息、`client_abort`、寫入對話紀錄、逐一呼叫 MCP 工具、
   `enqueue_tool_report`、組回覆、`speak_txt` 等流程。
2. **判斷方式單純依賴關鍵字字串比對**，容易誤觸發
   （例如「幫我」這個詞在「需要幫助/幫我」場景下會誤判成緊急呼叫，
   但在「幫我開燈」「幫我查空氣品質」裡也出現）。

本次重構分兩部分：

- **執行層收斂**：把四類獨立的硬解析函式收斂成「路由表 + 共用派發器」架構。
- **判斷層升級（燈光類別示範）**：把「這是不是燈光指令」的判斷，
  從關鍵字硬比對換成「語意相似度分類」，更貼近語意而非字面。

---

## 2. 執行層架構：路由表 + 共用派發器

### 2.1 資料結構

```python
@dataclass
class MatchResult:
    tools: list                              # [(工具名稱, 顯示用標籤), ...]
    reply: Union[str, Callable[[list], str]] # 固定回覆字串，或依工具結果動態組回覆的函式
    announce_before: Optional[str] = None    # 執行前要先講的話（例如 IR 學習模式提示）
    aggregate: bool = False                  # True = 多個工具結果合併成一句回覆
    missing_tools_message: str = "..."       # 工具不存在時要講的話

@dataclass
class IntentRoute:
    name: str                                # 路由識別名稱，方便寫 log
    match: Callable[[str], Optional[MatchResult]]  # 比對函式：文字 -> 命中結果或 None
    not_ready_message: str                   # func_handler 尚未初始化時的提示語
```

### 2.2 路由表

```python
INTENT_ROUTES = [
    IntentRoute("ir_remote",   match=match_ir_command,    not_ready_message="紅外線工具尚未初始化，請稍後再試。"),
    IntentRoute("sensor",      match=match_sensor_query,  not_ready_message="感測器工具尚未初始化，請稍後再試。"),
    IntentRoute("nurse_call",  match=match_nurse_command, not_ready_message="緊急呼叫工具尚未初始化，請稍後再試。"),
    IntentRoute("light",       match=match_light_command, not_ready_message="燈光控制還在準備中，請稍等一下。"),
]
```

順序即優先權：比對較精確的路由排前面，可避免「幫我」這類詞語被過早誤判成緊急呼叫。

### 2.3 共用派發器

```
dispatch_intent_routes(conn, text)
  └─ 依序呼叫每個 route.match(text)
       └─ 第一個傳回 MatchResult 的路由 → 交給 run_intent_route 執行

run_intent_route(conn, text, route, result)
  ├─ 檢查 conn.func_handler 是否就緒  → 沒有就講 route.not_ready_message
  ├─ 檢查 result.tools 裡每個工具是否存在 (has_tool)
  │    ├─ aggregate=False 且有缺 → 講 result.missing_tools_message，結束
  │    └─ aggregate=True  且全缺 → 講 result.missing_tools_message，結束
  ├─ 送出 STT 訊息 / 設定 client_abort / 寫入 dialogue
  ├─ 若有 announce_before → 先 speak_txt（例如 IR 學習模式提示）
  ├─ 逐一呼叫 conn.func_handler.handle_llm_function_call(...) 執行 MCP 工具
  │    └─ enqueue_tool_report 上報呼叫與結果
  ├─ aggregate=True  → 收集所有工具回覆，呼叫 result.reply(replies) 組成一句話
  └─ aggregate=False → 任一工具失敗就講錯誤訊息並結束；全部成功才講 result.reply
```

四類「比對函式」現在只需要回傳 `MatchResult`，原本各自的特例都對應到欄位上：

| 特例行為 | 對應欄位 |
|---|---|
| IR 學習模式要先講「請把遙控器對準接收器…」 | `announce_before` |
| 感測器查詢要把多個工具結果合併成一句話 | `aggregate=True` + `reply` 為 callable |
| 各類別工具不存在時的客製化提示 | `missing_tools_message` |

`handle_user_intent` 裡原本四個 `if await handle_hard_xxx(...)` 縮成一行：

```python
if await dispatch_intent_routes(conn, text):
    return True
```

---

## 3. 判斷層：燈光類別改用「語意相似度分類」

### 3.1 設計原則：分層判斷，各司其職

把燈光指令的判斷拆成兩個性質不同的子問題，分別用最適合的方法處理：

| 判斷項目 | 方法 | 理由 |
|---|---|---|
| **這句話是不是「燈光控制指令」** | 語意相似度分類 | 最容易誤判的地方：疑問句（「為什麼要開燈」）、否定句（「不用開燈了」）、單純聊天提到燈，字面上都含有「燈」這個字，純關鍵字比對無法分辨；語意分類能抓到「整句話的意思」而非「有沒有出現某個詞」 |
| 開燈還是關燈（動作） | 動詞詞表比對 | 動詞語意明確（「開」「打開」vs「關」「關閉」），字元級相似度（bigram）反而因為不分詞序，容易把「打開所有的燈」跟「關閉所有的燈」搞混（兩句字元重疊度極高） |
| 哪個顏色 / 是否全部 | 顏色詞表比對 | 顏色是明確實體（entity），屬於槽位萃取（slot extraction），不是意圖判斷 |

> 這個分層也是真實語意系統的常見做法：「intent classification」決定要不要做、
> 「slot filling」決定參數是什麼，兩者用的技術通常不同，硬要塞進同一個分類器
> 反而會讓彼此互相干擾。

### 3.2 語意相似度的實作方式

專案裡沒有安裝本地 embedding 模型（沒有 sentence-transformers 等套件），
呼叫遠端 embedding API 又會替每句話增加延遲與外部依賴。
因此採用**字元 bigram 向量 + cosine similarity** 做一個不需額外套件、
可離線運作的近似語意比對：

新增檔案：[`core/utils/semantic_similarity.py`](core/utils/semantic_similarity.py)

```python
def _bigram_vector(text):
    # 把字串切成相鄰兩字元一組（bigram），統計出現次數
    # 例如 "打开灯" → {"打开":1, "开灯":1}

def cosine_similarity(text_a, text_b):
    # 兩個 bigram 向量的餘弦相似度，0~1，越高越像

def best_match(text, examples: dict[label, list[範例句]]):
    # 對每個 label，取 text 與其所有範例句相似度的「最大值」
    # 回傳分數最高的 label 與其分數
```

這個比對的是「這句話跟範例句子像不像」，而不是「這句話裡有沒有出現某個關鍵字」，
因此可以承受語序、口語化、繁簡用字差異（只要範例庫涵蓋足夠的講法）。
若未來要換成真正的 embedding 模型或 API，只需保留 `best_match` 的介面，
原地替換內部實作即可，外層完全不用動。

### 3.3 範例庫設計

```python
_LIGHT_INTENT_EXAMPLES = {
    "light_command": [
        # 涵蓋「開」「關」「所有/單一顏色」「繁體 + 簡體」各種講法
        "幫我開燈", "把燈打開", "開燈", "燈光打開", "把綠燈打開", "紅燈開一下",
        "幫我關燈", "把燈關掉", "燈光關閉", "關閉所有的燈", ...
        "打开红灯", "打开绿灯", "关掉蓝灯", "关闭所有的灯", ...   # 簡體
    ],
    "not_light": [
        # 容易誤判的疑問句、否定句、無關聊天
        "為什麼要開燈", "不用開燈了", "今天天氣如何", "你叫什麼名字", ...
        "为什么要开灯", "不用开灯了", "今天天气如何", ...           # 簡體
    ],
}
_LIGHT_INTENT_THRESHOLD = 0.28
```

> **重要踩坑記錄**：第一版範例庫只收錄繁體中文，但 ASR 輸出的是簡體中文
> （例如「打开红灯」），bigram 是逐字元比對，繁簡字形不同（開≠开、燈≠灯），
> 相似度會被嚴重低估而判斷成「不是燈光指令」，導致整句話落到 LLM
> function-calling 路徑，產生「需要幫忙嗎？」之類的多餘對話與異常輸出。
> 修正方式：每一類範例都同時收錄繁體與簡體版本。

> **第二個踩坑**：曾經嘗試把 `light_command` 拆成 `light_open` / `light_close`
> 兩個 label 直接做相似度分類，結果「關閉所有的燈」被誤判成 `light_open`——
> 因為「打開所有的燈」「關閉所有的燈」字元重疊度極高（bag-of-bigram 不分詞序），
> 範例庫稍微不平衡就會讓某個方向「贏過頭」。最後改回：相似度只負責
> 「是不是燈光指令」這個二元判斷（`light_command` vs `not_light`），
> 開/關方向交還給語意明確的動詞詞表判斷，兩種方法各自處理它們最擅長的子問題。

### 3.4 完整判斷流程（以「打开红灯」為例）

```
ASR 文字: "打开红灯。"
normalized = "打开红灯"   # 去標點、轉小寫

① 否定詞守門（規則比對）
   negation_terms = ["不要","別","别","不用","不需要",...]
   命中就直接 return None（這是 bigram 相似度的天生弱點：
   「不用開燈」跟「開燈」字面很像，分數會偏高，需要規則守門先擋掉）
   → "打开红灯" 沒中，繼續

② 是不是「燈光指令」？── 語意相似度分類（核心判斷）
   label, score = best_match(text, _LIGHT_INTENT_EXAMPLES)
   → label = "light_command", score = 0.866
   判斷 label == "light_command" 且 score >= 0.28（門檻）→ 通過

③ 開或關？── 動詞詞表比對（槽位萃取）
   close_terms = ["關閉","关闭","關掉","关掉","關","关","熄滅","熄灭",...]
   open_terms  = ["打開","打开","開啟","开启","開","开","亮",...]
   "打开红灯" 含「打开」→ action = "open"

④ 哪個顏色？── 顏色詞表比對（槽位萃取）
   含 "红"/"紅" → colors = ["red"]

⑤ 是否要全部？
   all_terms = ["所有","全部","全都",...]
   沒命中 → use_all = False

⑥ 查表組成 MatchResult
   tool_map[("red","open")] = ("紅燈_打開", "紅燈")
   → MatchResult(
         tools=[("紅燈_打開", "紅燈")],
         reply="好的，紅燈已打開。",
         missing_tools_message="燈光工具還沒有準備好，請稍等一下再試。"
     )
```

接著回到共用派發器執行：

```
dispatch_intent_routes 取得 MatchResult
  → run_intent_route(conn, text, route, result)
       ├─ 確認 conn.func_handler 就緒、"紅燈_打開" 工具存在
       ├─ send_stt_message / dialogue.put / client_abort
       ├─ 呼叫 conn.func_handler.handle_llm_function_call(...) → 觸發 MCP 工具
       └─ speak_txt(conn, "好的，紅燈已打開。")
```

---

## 4. 測試結果

用實際 ASR 可能輸出的句子（含繁簡、否定句、無關問句）做了單元測試：

| 輸入 | 分類結果 | 動作 | 備註 |
|---|---|---|---|
| 关闭所有的灯。 | light_command (0.913) | close | 修正後正確判斷為「關」 |
| 打开红灯。 | light_command (0.866) | open | |
| 打开绿灯。 | light_command (0.866) | open | |
| 关掉蓝灯 | light_command (1.0) | close | |
| 打开所有的灯 | light_command (1.0) | open | |
| 帮我把绿灯关掉 | light_command (0.816) | close | |
| 为什么要开灯 | not_light → None | — | 正確擋掉，不觸發 |
| 不用开灯了 | 否定詞守門 → None | — | 正確擋掉 |
| 今天天气如何 | not_light → None | — | 正確擋掉 |
| 帮我查一下空气品质 | not_light → None | — | 正確判斷不是燈光指令（會交給感測器路由） |
| 房间有点闷 | not_light → None | — | 正確判斷不是燈光指令 |

並用 `py_compile` 確認語法正確、grep 確認沒有殘留對舊函式名稱的引用。

---

## 5. 異於「關鍵字硬解析」之處

```
✗ 硬解析：if "空氣品質" in text: call sensor
✓ 本系統：先用語意相似度判斷「這句話的整體意思像不像燈光指令」，
          再用語意明確的動詞/實體詞表萃取「要做什麼動作、對哪個對象」
```

差別在於：硬解析是「文字裡有沒有出現某個固定字串」，
一旦使用者換句話說（「房間有點悶」想查環境、卻被「燈」這個字誤觸發）
或用否定句、疑問句，就會誤判；
本系統是先問「這句話整體上是不是在下達某類指令」（語意分類），
true 之後才用詞表去抽取「動作、顏色」這些語意明確、不易混淆的槽位資訊，
關鍵字只用在「字義本身就很明確、不需要語境判斷」的子問題上。

---

## 6. 後續可擴充方向

- 其餘三類（紅外線、感測器、緊急呼叫）目前仍是關鍵字硬比對，
  若要採用相同的語意分類設計，只需：
  1. 為每一類整理「正例 / 反例」範例句（含繁簡）
  2. 用 `best_match` 判斷「是不是該類指令」
  3. 動作/參數萃取維持用詞表（槽位萃取）
- 若未來要換成真正的 embedding 模型或遠端 API，
  只需替換 `core/utils/semantic_similarity.py` 裡 `best_match` 的實作，
  外層 `match_*` / `INTENT_ROUTES` / `dispatch_intent_routes` 完全不用更動。
- 範例庫建議集中管理、定期用實際 log 裡的誤判案例補充，
  讓分類器持續校準（這也是語意分類相對關鍵字硬解析的優勢——
  改善方式是「補語料」而不是「改程式碼邏輯」）。
