import json
import uuid
import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.connection import ConnectionHandler
from core.utils.dialogue import Message
from core.providers.tts.dto.dto import ContentType
from core.handle.helloHandle import checkWakeupWords
from plugins_func.register import Action, ActionResponse
from core.handle.sendAudioHandle import send_stt_message
from core.handle.reportHandle import enqueue_tool_report
from core.utils.util import remove_punctuation_and_length
from core.providers.tts.dto.dto import TTSMessageDTO, SentenceType

TAG = __name__


async def handle_user_intent(conn: "ConnectionHandler", text):
    # 预处理输入文本，处理可能的JSON格式
    try:
        if text.strip().startswith("{") and text.strip().endswith("}"):
            parsed_data = json.loads(text)
            if isinstance(parsed_data, dict) and "content" in parsed_data:
                text = parsed_data["content"]  # 提取content用于意图分析
                conn.current_speaker = parsed_data.get("speaker")  # 保留说话人信息
    except (json.JSONDecodeError, TypeError):
        pass

    # 检查是否有明确的退出命令
    _, filtered_text = remove_punctuation_and_length(text)
    if await check_direct_exit(conn, filtered_text):
        return True

    # 检查是否是唤醒词
    if await checkWakeupWords(conn, filtered_text):
        return True

    # 燈光控制走硬解析，不交給大模型猜工具名
    if await handle_hard_ir_command(conn, text):
        return True

    if await handle_hard_sensor_query(conn, text):
        return True

    if await handle_hard_nurse_command(conn, text):
        return True

    if await handle_hard_light_command(conn, text):
        return True

    if conn.intent_type == "function_call":
        # 使用支持function calling的聊天方法,不再进行意图分析
        return False
    # 使用LLM进行意图分析
    intent_result = await analyze_intent_with_llm(conn, text)
    if not intent_result:
        return False
    # 会话开始时生成sentence_id
    conn.sentence_id = str(uuid.uuid4().hex)
    # 处理各种意图
    return await process_intent_result(conn, intent_result, text)


async def check_direct_exit(conn: "ConnectionHandler", text):
    """检查是否有明确的退出命令"""
    _, text = remove_punctuation_and_length(text)
    cmd_exit = conn.cmd_exit
    for cmd in cmd_exit:
        if text == cmd:
            conn.logger.bind(tag=TAG).info(f"识别到明确的退出命令: {text}")
            await send_stt_message(conn, text)
            await conn.close()
            return True
    return False


async def analyze_intent_with_llm(conn: "ConnectionHandler", text):
    """使用LLM分析用户意图"""
    if not hasattr(conn, "intent") or not conn.intent:
        conn.logger.bind(tag=TAG).warning("意图识别服务未初始化")
        return None

    # 对话历史记录
    dialogue = conn.dialogue
    try:
        intent_result = await conn.intent.detect_intent(conn, dialogue.dialogue, text)
        return intent_result
    except Exception as e:
        conn.logger.bind(tag=TAG).error(f"意图识别失败: {str(e)}")

    return None


async def process_intent_result(
    conn: "ConnectionHandler", intent_result, original_text
):
    """处理意图识别结果"""
    try:
        # 尝试将结果解析为JSON
        intent_data = json.loads(intent_result)

        # 检查是否有function_call
        if "function_call" in intent_data:
            # 直接从意图识别获取了function_call
            conn.logger.bind(tag=TAG).debug(
                f"检测到function_call格式的意图结果: {intent_data['function_call']['name']}"
            )
            function_name = intent_data["function_call"]["name"]
            if function_name == "continue_chat":
                return False

            if function_name == "result_for_context":
                await send_stt_message(conn, original_text)
                conn.client_abort = False

                def process_context_result():
                    conn.dialogue.put(Message(role="user", content=original_text))

                    from core.utils.current_time import get_current_time_info

                    current_time, today_date, today_weekday, lunar_date = (
                        get_current_time_info()
                    )

                    # 构建带上下文的基础提示
                    context_prompt = f"""当前时间：{current_time}
                                        今天日期：{today_date} ({today_weekday})
                                        今天农历：{lunar_date}

                                        请根据以上信息回答用户的问题：{original_text}"""

                    response = conn.intent.replyResult(context_prompt, original_text)
                    speak_txt(conn, response)

                conn.executor.submit(process_context_result)
                return True

            function_args = {}
            if "arguments" in intent_data["function_call"]:
                function_args = intent_data["function_call"]["arguments"]
                if function_args is None:
                    function_args = {}
            # 确保参数是字符串格式的JSON
            if isinstance(function_args, dict):
                function_args = json.dumps(function_args)

            function_call_data = {
                "name": function_name,
                "id": str(uuid.uuid4().hex),
                "arguments": function_args,
            }

            await send_stt_message(conn, original_text)
            conn.client_abort = False

            # 准备工具调用参数
            tool_input = {}
            if function_args:
                if isinstance(function_args, str):
                    tool_input = json.loads(function_args) if function_args else {}
                elif isinstance(function_args, dict):
                    tool_input = function_args

            # 上报工具调用
            enqueue_tool_report(conn, function_name, tool_input)

            # 使用executor执行函数调用和结果处理
            def process_function_call():
                conn.dialogue.put(Message(role="user", content=original_text))
                
                # 工具调用超时时间
                tool_call_timeout = int(conn.config.get("tool_call_timeout", 30))
                # 使用统一工具处理器处理所有工具调用
                try:
                    result = asyncio.run_coroutine_threadsafe(
                        conn.func_handler.handle_llm_function_call(
                            conn, function_call_data
                        ),
                        conn.loop,
                    ).result(timeout=tool_call_timeout)
                except Exception as e:
                    conn.logger.bind(tag=TAG).error(f"工具调用失败: {e}")
                    result = ActionResponse(
                        action=Action.ERROR, result="工具调用超时，请一会再试下哈", response="工具调用超时，请一会再试下哈"
                    )

                # 上报工具调用结果
                if result:
                    enqueue_tool_report(conn, function_name, tool_input, str(result.result) if result.result else None, report_tool_call=False)

                    if result.action == Action.RESPONSE:  # 直接回复前端
                        text = result.response
                        if text is not None:
                            speak_txt(conn, text)
                    elif result.action == Action.REQLLM:  # 调用函数后再请求llm生成回复
                        text = result.result
                        conn.dialogue.put(Message(role="tool", content=text))
                        llm_result = conn.intent.replyResult(text, original_text)
                        if llm_result is None:
                            llm_result = text
                        speak_txt(conn, llm_result)
                    elif (
                        result.action == Action.NOTFOUND
                        or result.action == Action.ERROR
                    ):
                        text = result.response if result.response else result.result
                        if text is not None:
                            speak_txt(conn, text)
                    elif function_name != "play_music":
                        # For backward compatibility with original code
                        # 获取当前最新的文本索引
                        text = result.response
                        if text is None:
                            text = result.result
                        if text is not None:
                            speak_txt(conn, text)

            # 将函数执行放在线程池中
            conn.executor.submit(process_function_call)
            return True
        return False
    except json.JSONDecodeError as e:
        conn.logger.bind(tag=TAG).error(f"处理意图结果时出错: {e}")
        return False




def parse_hard_ir_command(text):
    if not isinstance(text, str):
        return None

    normalized = (
        text.lower()
        .replace(" ", "")
        .replace("，", "")
        .replace("。", "")
        .replace("！", "")
        .replace("？", "")
        .replace("!", "")
        .replace("?", "")
    )

    learn = any(term in normalized for term in [
        "學習", "学习", "記住", "记住", "配對", "配对", "錄製", "录制", "learn"
    ])

    status_terms = ["狀態", "状态", "學過", "学过", "記憶", "记忆", "status"]
    if any(term in normalized for term in status_terms) and any(term in normalized for term in ["紅外線", "红外线", "ir", "風扇", "风扇", "電風扇", "电风扇", "關燈器", "关灯器"]):
        return ("self_ir_status", "紅外線狀態")

    fan_terms = ["電風扇", "电风扇", "風扇", "风扇", "fan"]
    light_switch_terms = ["關燈器", "关灯器", "遙控燈", "遥控灯", "燈控", "灯控", "light"]

    wants_fan = any(term in normalized for term in fan_terms)
    wants_light_switch = any(term in normalized for term in light_switch_terms)

    if wants_fan:
        speed_down_terms = ["調弱", "调弱", "弱", "變弱", "变弱", "減弱", "减弱", "降低", "降速", "慢一點", "慢一点", "speeddown", "down", "-"]
        if any(term in normalized for term in speed_down_terms):
            return ("self_ir_learn_fan_speed_down" if learn else "self_ir_send_fan_speed_down", "電風扇調弱")

        speed_up_terms = ["加強", "加强", "強", "强", "變強", "变强", "增強", "增强", "提高", "升速", "快一點", "快一点", "speedup", "up", "+"]
        general_speed_terms = ["強弱", "强弱", "風速", "风速", "速度", "調速", "调速", "檔位", "档位"]
        if any(term in normalized for term in speed_up_terms) or any(term in normalized for term in general_speed_terms):
            return ("self_ir_learn_fan_speed" if learn else "self_ir_send_fan_speed", "電風扇加強")

        power_terms = ["開關", "开关", "開", "开", "關", "关", "電源", "电源", "啟動", "启动", "停止", "power", "on", "off"]
        if learn or any(term in normalized for term in power_terms):
            return ("self_ir_learn_fan_power" if learn else "self_ir_send_fan_power", "電風扇開關")

    if wants_light_switch:
        power_terms = ["開關", "开关", "開", "开", "關", "关", "電源", "电源", "power", "on", "off"]
        if learn or any(term in normalized for term in power_terms):
            return ("self_ir_learn_light_power" if learn else "self_ir_send_light_power", "關燈器開關")

    return None
async def handle_hard_ir_command(conn: "ConnectionHandler", text):
    parsed = parse_hard_ir_command(text)
    if not parsed:
        return False

    tool_name, label = parsed
    if not conn.func_handler:
        conn.sentence_id = str(uuid.uuid4().hex)
        await send_stt_message(conn, text)
        speak_txt(conn, "紅外線工具尚未初始化，請稍後再試。")
        return True

    if not conn.func_handler.has_tool(tool_name):
        conn.sentence_id = str(uuid.uuid4().hex)
        await send_stt_message(conn, text)
        speak_txt(conn, "目前找不到紅外線工具，請確認設備已重新燒錄並連線。")
        return True

    conn.logger.bind(tag=TAG).info(f"硬解析紅外線風扇指令: {text} -> {tool_name}")
    conn.sentence_id = str(uuid.uuid4().hex)
    await send_stt_message(conn, text)
    conn.client_abort = False
    conn.dialogue.put(Message(role="user", content=text))

    if "learn" in tool_name:
        speak_txt(conn, f"請把風扇遙控器對準紅外線接收器，按下{label}按鍵。")

    function_call_data = {
        "name": tool_name,
        "id": str(uuid.uuid4().hex),
        "arguments": "{}",
    }
    enqueue_tool_report(conn, tool_name, {})
    result = await conn.func_handler.handle_llm_function_call(conn, function_call_data)
    enqueue_tool_report(
        conn,
        tool_name,
        {},
        str(result.result) if result and result.result else None,
        report_tool_call=False,
    )

    if result and result.action in [Action.ERROR, Action.NOTFOUND]:
        reply = result.response if result.response else result.result
    elif result:
        reply = result.response if result.response else result.result
    else:
        reply = "紅外線工具沒有回傳結果。"

    speak_txt(conn, str(reply))
    return True

def parse_hard_sensor_query(text):
    if not isinstance(text, str):
        return None

    normalized = (
        text.lower()
        .replace(" ", "")
        .replace("，", "")
        .replace("。", "")
        .replace("！", "")
        .replace("!", "")
        .replace("?", "")
        .replace("？", "")
    )

    air_terms = [
        "空氣品質", "空气品质", "空氣質量", "空气质量", "空氣", "空气",
        "tvoc", "voc", "甲醛", "異味", "异味",
    ]
    temp_terms = [
        "溫溼度", "温湿度", "溫濕度", "温濕度", "溫度", "温度",
        "濕度", "湿度", "dht", "幾度", "几度",
    ]

    wants_air = any(term in normalized for term in air_terms)
    wants_temp = any(term in normalized for term in temp_terms)

    if wants_air and wants_temp:
        return [
            ("空氣品質_讀取", "空氣品質"),
            ("溫溼度_讀取", "溫溼度"),
        ]
    if wants_air:
        return [("空氣品質_讀取", "空氣品質")]
    if wants_temp:
        return [("溫溼度_讀取", "溫溼度")]
    return None


async def handle_hard_sensor_query(conn: "ConnectionHandler", text):
    parsed = parse_hard_sensor_query(text)
    if not parsed:
        return False

    if not conn.func_handler:
        conn.sentence_id = str(uuid.uuid4().hex)
        await send_stt_message(conn, text)
        speak_txt(conn, "感測器工具尚未初始化，請稍後再試。")
        return True

    available_tools = []
    missing_labels = []
    for tool_name, label in parsed:
        if conn.func_handler.has_tool(tool_name):
            available_tools.append((tool_name, label))
        else:
            missing_labels.append(label)

    if not available_tools:
        conn.sentence_id = str(uuid.uuid4().hex)
        await send_stt_message(conn, text)
        speak_txt(conn, "目前找不到感測器讀取工具，請確認設備 MCP 已連線。")
        return True

    conn.logger.bind(tag=TAG).info(f"硬解析感測器查詢: {text} -> {[name for name, _ in available_tools]}")
    conn.sentence_id = str(uuid.uuid4().hex)
    await send_stt_message(conn, text)
    conn.client_abort = False
    conn.dialogue.put(Message(role="user", content=text))

    replies = []
    for tool_name, _ in available_tools:
        function_call_data = {
            "name": tool_name,
            "id": str(uuid.uuid4().hex),
            "arguments": "{}",
        }
        enqueue_tool_report(conn, tool_name, {})
        result = await conn.func_handler.handle_llm_function_call(conn, function_call_data)
        enqueue_tool_report(
            conn,
            tool_name,
            {},
            str(result.result) if result and result.result else None,
            report_tool_call=False,
        )

        if result and result.action in [Action.ERROR, Action.NOTFOUND]:
            replies.append(result.response if result.response else result.result)
        elif result:
            replies.append(result.response if result.response else result.result)

    replies = [str(reply) for reply in replies if reply]
    if missing_labels:
        replies.append("另外，" + "、".join(missing_labels) + "工具目前沒有連上。")

    speak_txt(conn, " ".join(replies) if replies else "感測器沒有回傳資料，請稍後再試。")
    return True

def parse_hard_nurse_command(text):
    if not isinstance(text, str):
        return None

    normalized = (
        text.lower()
        .replace(" ", "")
        .replace("，", "")
        .replace("。", "")
        .replace("！", "")
        .replace("!", "")
        .replace("?", "")
        .replace("？", "")
    )

    cancel_terms = [
        "取消報警", "取消报警", "停止報警", "停止报警", "關閉報警", "关闭报警",
        "取消蜂鳴器", "取消蜂鸣器", "停止蜂鳴器", "停止蜂鸣器", "關閉蜂鳴器", "关闭蜂鸣器",
        "不用幫助", "不用帮助", "解除緊急", "解除紧急",
    ]
    trigger_terms = [
        "報警", "报警", "蜂鳴器", "蜂鸣器", "呼叫蜂鳴器", "呼叫蜂鸣器",
        "需要幫助", "需要帮助", "幫我", "帮我", "救命", "緊急", "紧急",
        "急救", "求救", "呼叫護士", "呼叫护士", "護理師", "护理师",
    ]

    if any(term in normalized for term in cancel_terms):
        return "self_nurse_call_cancel", "已取消緊急呼叫。"
    if any(term in normalized for term in trigger_terms):
        return "self_nurse_call_trigger", "已啟動緊急呼叫，需要幫助。"
    return None


async def handle_hard_nurse_command(conn: "ConnectionHandler", text):
    parsed = parse_hard_nurse_command(text)
    if not parsed:
        return False

    tool_name, reply = parsed

    if not conn.func_handler:
        conn.sentence_id = str(uuid.uuid4().hex)
        await send_stt_message(conn, text)
        speak_txt(conn, "緊急呼叫工具尚未初始化，請稍後再試。")
        return True

    if not conn.func_handler.has_tool(tool_name):
        conn.logger.bind(tag=TAG).warning(f"Emergency tool missing: {tool_name}")
        conn.sentence_id = str(uuid.uuid4().hex)
        await send_stt_message(conn, text)
        speak_txt(conn, "目前找不到緊急呼叫工具，請確認設備已連線。")
        return True

    conn.logger.bind(tag=TAG).info(f"硬解析緊急呼叫: {text} -> {tool_name}")

    conn.sentence_id = str(uuid.uuid4().hex)
    await send_stt_message(conn, text)
    conn.client_abort = False
    conn.dialogue.put(Message(role="user", content=text))

    function_call_data = {
        "name": tool_name,
        "id": str(uuid.uuid4().hex),
        "arguments": "{}",
    }

    enqueue_tool_report(conn, tool_name, {})
    result = await conn.func_handler.handle_llm_function_call(conn, function_call_data)
    enqueue_tool_report(
        conn,
        tool_name,
        {},
        str(result.result) if result and result.result else None,
        report_tool_call=False,
    )

    if result and result.action in [Action.ERROR, Action.NOTFOUND]:
        error_text = result.response if result.response else result.result
        speak_txt(conn, error_text or "緊急呼叫執行失敗，請再試一次。")
        return True

    speak_txt(conn, reply)
    return True

def parse_hard_light_command(text):
    if not isinstance(text, str):
        return None

    normalized = text.lower().replace(" ", "").replace("，", "").replace("。", "")

    if not any(word in normalized for word in ["燈", "灯", "light"]):
        return None

    if any(word in normalized for word in ["為什麼", "为什么", "怎麼", "怎么", "如何", "什麼", "什么"]):
        return None

    if any(word in normalized for word in ["不要", "別", "别", "不用", "不需要"]):
        return None

    close_terms = ["關閉", "关闭", "關掉", "关掉", "關", "关", "熄滅", "熄灭", "熄", "off", "close"]
    open_terms = ["打開", "打开", "開啟", "开启", "開", "开", "亮", "打亮", "on", "open"]

    if any(word in normalized for word in close_terms):
        action = "close"
    elif any(word in normalized for word in open_terms):
        action = "open"
    else:
        return None

    colors = []
    if any(word in normalized for word in ["綠", "绿", "green"]):
        colors.append("green")
    if any(word in normalized for word in ["紅", "红", "red"]):
        colors.append("red")
    if any(word in normalized for word in ["藍", "蓝", "blue"]):
        colors.append("blue")

    all_terms = ["所有", "全部", "全都", "三個", "三个", "全部的", "所有的", "另外", "剩下"]
    use_all = any(word in normalized for word in all_terms)

    tool_map = {
        ("green", "open"): ("綠燈_打開", "綠燈"),
        ("green", "close"): ("綠燈_關閉", "綠燈"),
        ("red", "open"): ("紅燈_打開", "紅燈"),
        ("red", "close"): ("紅燈_關閉", "紅燈"),
        ("blue", "open"): ("藍燈_打開", "藍燈"),
        ("blue", "close"): ("藍燈_關閉", "藍燈"),
    }

    if use_all or not colors:
        tool_name = "所有灯_打开" if action == "open" else "所有灯_关闭"
        reply = "好的，所有燈已打開。" if action == "open" else "好的，所有燈已關閉。"
        return [tool_name], reply

    tools = []
    labels = []
    for color in colors:
        tool_name, label = tool_map[(color, action)]
        tools.append(tool_name)
        labels.append(label)

    action_text = "打開" if action == "open" else "關閉"
    reply = f"好的，{'、'.join(labels)}已{action_text}。"
    return tools, reply


async def handle_hard_light_command(conn: "ConnectionHandler", text):
    parsed = parse_hard_light_command(text)
    if not parsed:
        return False

    tool_names, reply = parsed

    if not conn.func_handler:
        conn.sentence_id = str(uuid.uuid4().hex)
        await send_stt_message(conn, text)
        speak_txt(conn, "燈光控制還在準備中，請稍等一下。")
        return True

    missing_tools = [name for name in tool_names if not conn.func_handler.has_tool(name)]
    if missing_tools:
        conn.logger.bind(tag=TAG).warning(f"硬解析命中，但工具不存在: {missing_tools}")
        conn.sentence_id = str(uuid.uuid4().hex)
        await send_stt_message(conn, text)
        speak_txt(conn, "燈光工具還沒有準備好，請稍等一下再試。")
        return True

    conn.logger.bind(tag=TAG).info(f"硬解析燈光指令: {text} -> {tool_names}")

    conn.sentence_id = str(uuid.uuid4().hex)
    await send_stt_message(conn, text)
    conn.client_abort = False
    conn.dialogue.put(Message(role="user", content=text))

    for tool_name in tool_names:
        function_call_data = {
            "name": tool_name,
            "id": str(uuid.uuid4().hex),
            "arguments": "{}",
        }

        enqueue_tool_report(conn, tool_name, {})

        result = await conn.func_handler.handle_llm_function_call(conn, function_call_data)
        enqueue_tool_report(
            conn,
            tool_name,
            {},
            str(result.result) if result and result.result else None,
            report_tool_call=False,
        )

        if result and result.action in [Action.ERROR, Action.NOTFOUND]:
            error_text = result.response if result.response else result.result
            speak_txt(conn, error_text or "燈光控制失敗，請再試一次。")
            return True

    speak_txt(conn, reply)
    return True

def speak_txt(conn: "ConnectionHandler", text):
    # 记录文本到 sentence_id 映射
    conn.tts.store_tts_text(conn.sentence_id, text)

    conn.tts.tts_text_queue.put(
        TTSMessageDTO(
            sentence_id=conn.sentence_id,
            sentence_type=SentenceType.FIRST,
            content_type=ContentType.ACTION,
        )
    )
    conn.tts.tts_one_sentence(conn, ContentType.TEXT, content_detail=text)
    conn.tts.tts_text_queue.put(
        TTSMessageDTO(
            sentence_id=conn.sentence_id,
            sentence_type=SentenceType.LAST,
            content_type=ContentType.ACTION,
        )
    )
    conn.dialogue.put(Message(role="assistant", content=text))
