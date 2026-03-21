import base64
import os
from typing import Dict, Any
from urllib.parse import quote

import pandas as pd
import streamlit as st
import requests
import json

# DeepSeek API（复用现有调用；避免在多个函数里重复写 Key）
DEEPSEEK_API_KEY = "sk-cfd82383a0e645119e37bb9a05200883"


@st.cache_data
def load_mcq_bank(path: str) -> pd.DataFrame:
    """
    加载选择题题库 PBL题库.xlsx。
    如果文件不存在，则在页面上显示错误并停止运行。
    """
    if not os.path.exists(path):
        st.error(f"未找到选择题题库文件：{path}")
        st.stop()
    try:
        df = pd.read_excel(path)
    except Exception as e:
        st.error(f"加载选择题题库时出错：{e}")
        st.stop()
    return df



def load_short_question_bank(path: str) -> pd.DataFrame:
    """
    加载简答题题库 简答题题库.xlsx。
    如果文件不存在，则在页面上显示错误并停止运行。
    """
    if not os.path.exists(path):
        st.error(f"未找到简答题题库文件：{path}")
        st.stop()
    try:
        df = pd.read_excel(path)
    except Exception as e:
        st.error(f"加载简答题题库时出错：{e}")
        st.stop()
    return df


@st.cache_data
def load_short_question_scoring(path: str) -> pd.DataFrame:
    """
    加载简答题评分规则 简答题评分.xlsx。
    如果文件不存在，则在页面上显示错误并停止运行。
    """
    if not os.path.exists(path):
        st.error(f"未找到简答题评分文件：{path}")
        st.stop()
    try:
        df = pd.read_excel(path)
    except Exception as e:
        st.error(f"加载简答题评分文件时出错：{e}")
        st.stop()
    return df


@st.cache_data
def load_patient_profiles(path: str) -> pd.DataFrame:
    """加载病人设定文件 病人设定.xlsx。"""
    if not os.path.exists(path):
        st.error(f"未找到病人设定文件：{path}")
        st.stop()
    try:
        df = pd.read_excel(path)
        return df
    except Exception as e:
        st.error(f"加载病人设定文件时出错：{e}")
        st.stop()


@st.cache_data
def load_scene_narrator(path: str) -> pd.DataFrame:
    """加载情景交代文件 情景交代.xlsx。如果文件不存在，返回空 DataFrame 并显示警告。"""
    if not os.path.exists(path):
        st.warning(f"未找到情景交代文件：{path}，将使用默认情景交代。")
        return pd.DataFrame()
    try:
        df = pd.read_excel(path)
        df.columns = df.columns.str.strip()  # 去除列名空格
        return df
    except Exception as e:
        st.warning(f"加载情景交代文件时出错：{e}，将使用默认情景交代。")
        return pd.DataFrame()


@st.cache_data
def load_teaching_outlines(path: str) -> pd.DataFrame:
    """
    加载医小星教学大纲文件 医小星教学大纲.xlsx。
    如果文件不存在或加载失败，则返回空 DataFrame，并给出警告。
    该文件中建议包含列：疾病、情境、阶段（如“症状分析”/“背景知识”）、教学大纲。
    """
    if not os.path.exists(path):
        st.warning(f"未找到医小星教学大纲文件：{path}，将使用内置默认教学大纲。")
        return pd.DataFrame()
    try:
        df = pd.read_excel(path)
        df.columns = df.columns.str.strip()  # 去除列名空格
        return df
    except Exception as e:
        st.warning(f"加载医小星教学大纲时出错，将使用内置默认教学大纲：{e}")
        return pd.DataFrame()


@st.cache_data
def load_case_outlines(path: str) -> pd.DataFrame:
    """加载医小星深度精讲表 `data/疾病精讲.xlsx`（系统、疾病、步骤与提示词模板）。"""
    if not os.path.exists(path):
        st.error(f"未找到医小星深度精讲配置文件：{path}")
        st.stop()
    try:
        df = pd.read_excel(path)
        df.columns = df.columns.str.strip()
        required_cols = ["系统", "疾病", "步骤序号", "步骤名称", "讲解提示词模板"]
        for col in required_cols:
            if col not in df.columns:
                st.error(f"配置文件缺少列：{col}")
                st.stop()
        # 规范化文本列，避免 Excel 数字/空格导致筛选与 session 不一致（如 42 疾病 × 多步）
        for c in ("系统", "疾病", "步骤名称", "讲解提示词模板"):
            if c in df.columns:
                df[c] = df[c].apply(lambda x: str(x).strip() if pd.notna(x) else "")
        if "步骤序号" in df.columns:
            df["步骤序号"] = pd.to_numeric(df["步骤序号"], errors="coerce")
        df = df.dropna(subset=["步骤序号"])
        return df
    except Exception as e:
        st.error(f"加载配置文件出错：{e}")
        st.stop()


def call_ai_patient(messages, profile):
    """使用 DeepSeek API 生成 AI 患者回复"""
    api_key = DEEPSEEK_API_KEY

    # 从病人设定中安全获取字段
    def get_field(name: str) -> str:
        if name in profile.index:
            val = profile.get(name, "")
            if pd.notna(val):
                return str(val)
        return ""

    system_prompt = f"""
你是一位{get_field('年龄')}岁的{get_field('性别')}性患者。
当前阶段：{get_field('阶段描述')}
你的症状：{get_field('症状')}
体征：{get_field('体征')}
辅助检查：{get_field('辅助检查')}
既往史：{get_field('既往史')}
性格：{get_field('性格')}
对话风格：{get_field('对话风格')}
你对自己病情的认知：{get_field('当前认知')}
你最关心的问题：{get_field('关心问题')}

规则：
1. 你只能回答医生问到的问题，不要主动提供额外信息（但如果是开场白，可以主动说第一句话）。
2. 用口语化表达，像真实患者一样。
3. 保持人设一致性。
"""
    full_messages = [{"role": "system", "content": system_prompt}] + messages

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "deepseek-chat",
        "messages": full_messages,
        "temperature": 0.7,
        "max_tokens": 500,
    }

    # 获取代理设置（从环境变量）
    proxies = None
    http_proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
    https_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if http_proxy or https_proxy:
        proxies = {"http": http_proxy, "https": https_proxy}

    try:
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
            proxies=proxies,
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"【AI 患者暂时无法回答，错误：{str(e)}】"


def call_guide_assistant(
    patient_profile,
    patient_recent: str,
    guide_history: list,
    situation,
    student_level: str = "初级",
) -> str:
    """
    引导助手：根据患者最新回答和引导历史，按步骤引导学生学习。
    - patient_profile: 病人设定 (pd.Series)
    - patient_recent: 患者最新回答的文本
    - guide_history: 引导对话历史列表，每条消息格式 {"role": "user"/"assistant", "content": "..."}
    - situation: 当前情境编号 (1/2/3/4)
    - student_level: "初级" 或 "高级"
    """
    api_key = DEEPSEEK_API_KEY

    def get_field(name: str) -> str:
        if name in patient_profile.index:
            val = patient_profile.get(name, "")
            if pd.notna(val):
                return str(val)
        return ""

    # 确保 situation 为整数 1~4，供教学重点和大纲使用
    try:
        sit_num = int(situation) if situation is not None else 1
    except (TypeError, ValueError):
        sit_num = 1
    if sit_num < 1 or sit_num > 4:
        sit_num = min(max(sit_num, 1), 4)

    # 根据情境调整教学重点
    situation_guide = {
        1: "重点是引导学生捕捉症状关键词、结合背景信息分析病因，并推荐初步检查。",
        2: "重点是引导学生理解检查结果（CT、血脂等）的意义，解释脑出血的诊断依据。",
        3: "重点是引导学生讨论病情严重程度、治疗方法和预后，并用通俗语言向患者解释。",
        4: "重点是引导学生给出综合出院建议（用药、康复、生活方式），强调二级预防。",
    }

    # 构建教学大纲
    teaching_steps = """
1. **捕捉关键信息**：首先引导学生从患者的描述中找出关键信息（如头痛、右肢无力、说话不利索等）。
2. **结合背景信息**：然后引导学生关注患者的背景信息（年龄、吸烟史、饮酒史、高血压病史、未规律服药等），并提问这些信息对诊断有何帮助。
3. **高血压发病机制**：提问学生高血压的发病机制是什么，为什么会导致脑出血。
4. **其他可能信息**：提问学生还能从患者描述中捕捉到哪些其他有用信息（如工作压力大、应酬多等）。
5. **推荐辅助检查**：提问学生应该为患者推荐哪些辅助检查（如CT、MRI、血液检查等），以及为什么。
6. **完成引导**：在完成以上所有环节后，总结并提示学生可以返回问诊界面继续问诊。
"""

    system_prompt = f"""
你是一位医学教育引导助手，正在帮助医学生分析患者的回答并学习临床知识。
当前情境：情境{sit_num}
{situation_guide.get(sit_num, situation_guide[1])}

患者刚刚说：“{patient_recent}”
患者背景信息：年龄{get_field('年龄')}岁，性别{get_field('性别')}，既往史：{get_field('既往史')}，性格：{get_field('性格')}。

你的教学任务：按照以下大纲逐步引导学生学习，确保学生在每个步骤中都能理解并回答正确。如果学生回答正确或接近正确，给予肯定并进入下一步；如果学生不清楚，用通俗易懂的方式解释后再进入下一步。当所有步骤完成后，在最后一条回复中加上一句提示：“很好，你已经掌握了该情境的关键知识，现在可以返回问诊界面继续和患者对话啦！”

教学大纲：
{teaching_steps}

注意：
- 用鼓励、亲切的语气。
- 每次只提一个问题，等待学生回答后再继续。
- 根据学生的回答动态调整，不要一次性给出所有答案。
- 保持对话的连贯性。
"""

    full_messages = [{"role": "system", "content": system_prompt}] + list(guide_history)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "deepseek-chat",
        "messages": full_messages,
        "temperature": 0.7,
        "max_tokens": 500,
    }

    try:
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"【引导助手暂时无法回答，错误：{str(e)}】"


def call_medstar(
    guide_history: list,
    context: dict,
    student_level: str = "初级",
) -> str:
    """
    医小星引导助手：根据当前引导历史和上下文，生成下一步引导。
    - guide_history: 引导对话历史（列表，元素为 {"role": "user"/"assistant", "content": "..."}）
    - context: 字典，包含 profile（病例设定）、situation（情境编号），以及可选的 patient_recent（患者最新消息）；
      若存在 patient_recent 则为症状分析阶段，否则为背景知识教学阶段。
    - student_level: "初级" 或 "高级"
    返回医小星的回复字符串，教学完成时可在末尾加 [DONE] 标记。
    """
    api_key = DEEPSEEK_API_KEY

    profile = context.get("profile")
    situation = context.get("situation", 1)
    patient_recent = context.get("patient_recent", None)  # 可能为空，无则进入背景教学

    if profile is None:
        return "【医小星暂时无法回答：缺少病例设定】"

    def get_field(name: str) -> str:
        if name in profile.index:
            val = profile.get(name, "")
            if pd.notna(val):
                return str(val)
        return ""

    try:
        sit_num = int(situation) if situation is not None else 1
    except (TypeError, ValueError):
        sit_num = 1
    if sit_num < 1 or sit_num > 4:
        sit_num = min(max(sit_num, 1), 4)

    # 获取疾病名称，用于从外部大纲表中按疾病筛选
    disease_name = get_field("疾病")
    if disease_name is None:
        disease_name = ""
    disease_name = str(disease_name).strip()

    def _get_teaching_from_outlines(stage: str) -> str:
        """
        从 st.session_state.teaching_outlines 中按 疾病 / 情境 / 阶段 获取教学大纲文本。
        - stage: "症状分析" 或 "背景知识"
        若未找到或表为空，则返回空字符串，交由内置大纲兜底。
        """
        df_out = st.session_state.get("teaching_outlines")
        if not isinstance(df_out, pd.DataFrame) or df_out.empty:
            return ""
        required_cols = ["疾病", "情境", "阶段类型", "教学大纲"]
        if not all(col in df_out.columns for col in required_cols):
            return ""
        try:
            df_filtered = df_out[
                (df_out["疾病"] == disease_name)
                & (df_out["阶段类型"] == stage)
                & (df_out["情境"].astype(str) == str(sit_num))
            ]
        except Exception:
            return ""
        if df_filtered.empty:
            return ""
        text = df_filtered.iloc[0]["教学大纲"]
        if pd.isna(text):
            return ""
        return str(text).strip()

    if patient_recent:
        # 症状分析阶段：优先从 session_state 中的大纲表读取，若无则使用默认提示
        outline_text = _get_teaching_from_outlines("症状分析")
        if outline_text:
            teaching_outline = outline_text
        else:
            teaching_outline = "请根据病例引导学习。"

        system_prompt = f"""
你是一位名叫“医小星”的医学教育引导助手，说话亲切可爱，喜欢用✨和🎓等表情，帮助医学生分析患者信息并学习临床知识。
当前情境：情境{sit_num}。

患者刚刚说：“{patient_recent}”
患者背景：年龄{get_field('年龄')}岁，性别{get_field('性别')}，既往史：{get_field('既往史')}，性格：{get_field('性格')}。

现在进入**症状分析阶段**，{teaching_outline}

规则：
- 用鼓励、亲切的语气，可以适当使用表情符号。
- **每次只提一个问题**，不要一次性列出所有问题。
- 根据学生的回答动态决定下一步问题。如果学生答得很好，可以提前进入下一步。
- 若引导历史为空，则直接抛出第一步的问题。
- 当所有步骤完成后，在最后一条回复的末尾加上 `[DONE]`。
"""
    else:
        # 背景知识教学阶段：优先从 session_state 中的大纲表读取，若无则使用默认提示
        outline_text = _get_teaching_from_outlines("背景知识")
        if outline_text:
            teaching_content = outline_text
        else:
            teaching_content = "请根据病例引导学习。"

        system_prompt = f"""
你是一位名叫“医小星”的医学教育引导助手，说话亲切可爱，喜欢用✨和🎓等表情，帮助医学生分析患者信息并学习临床知识。
当前情境：情境{sit_num}。
患者背景：年龄{get_field('年龄')}岁，性别{get_field('性别')}，既往史：{get_field('既往史')}，性格：{get_field('性格')}。

现在进入**背景知识教学阶段**，{teaching_content}

规则：
- 用鼓励、亲切的语气，可以适当使用表情符号。
- **每次只提一个问题**，不要一次性列出所有问题。
- 根据学生的回答动态决定下一步问题。
- 若引导历史为空，则直接抛出第一步的问题。
- 当所有步骤完成后，在最后一条回复的末尾加上 `[DONE]`。
"""

    full_messages = [{"role": "system", "content": system_prompt}] + list(guide_history)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "deepseek-chat",
        "messages": full_messages,
        "temperature": 0.7,
        "max_tokens": 500,
    }

    # 获取代理设置（从环境变量）
    proxies = None
    http_proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
    https_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if http_proxy or https_proxy:
        proxies = {"http": http_proxy, "https": https_proxy}

    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
            verify=False,
            proxies=proxies,
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"【医小星暂时无法回答，错误：{str(e)}】"


def call_general_ai(user_message: str, history: list[dict[str, str]] | None = None) -> str:
    """调用 DeepSeek API 回答通用医学问题（不带病例上下文）。"""
    api_key = DEEPSEEK_API_KEY
    if not api_key:
        return "【医小星暂时无法回答：未配置 DeepSeek API Key】"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    system_msg = {
        "role": "system",
        "content": "你是一位医学教育助手，名叫医小星，亲切可爱，用通俗易懂的语言回答医学问题。"
        "如果涉及严重症状或紧急情况，请提醒用户及时就医或拨打急救电话。"
        "不要编造诊断；给出可操作的科普与就医建议。",
    }
    messages = [system_msg]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    payload = {
        "model": "deepseek-chat",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 500,
    }

    # 获取代理设置（从环境变量）
    proxies = None
    http_proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
    https_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if http_proxy or https_proxy:
        proxies = {"http": http_proxy, "https": https_proxy}

    try:
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
            proxies=proxies,
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"【医小星暂时无法回答，错误：{str(e)}】"


def init_session_state() -> None:
    """初始化全局 session_state，用于页面导航和学习状态管理。"""
    if "current_page" not in st.session_state:
        st.session_state.current_page = "home"
    if "current_system" not in st.session_state:
        st.session_state.current_system = None
    if "current_disease" not in st.session_state:
        st.session_state.current_disease = None
    if "grade" not in st.session_state:
        # 当前年级，后续如需切换可在此基础上扩展
        st.session_state.grade = "初级"
    if "situation_state" not in st.session_state:
        # 结构：
        # {
        #   情境值: {
        #       "q_index": int,
        #       "answers": {idx: "A"/"B"/...},
        #       "submitted": set(),
        #       "correct": {idx: bool},
        #       "knowledge": {idx: str},
        #       "essay_answer": str,
        #       "essay_submitted": bool,
        #       "essay_score": float,
        #       "essay_max": float,
        #       "essay_missing": list[str],
        #       "has_essay": bool,
        #   }
        # }
        st.session_state.situation_state: Dict[Any, Dict[str, Any]] = {}
    if "guide_mode" not in st.session_state:
        st.session_state.guide_mode = False  # 是否处于医小星引导页面
    if "guide_messages" not in st.session_state:
        st.session_state.guide_messages = []  # 医小星引导对话历史
    if "guide_context" not in st.session_state:
        st.session_state.guide_context = {}  # 当前引导所需上下文（患者最新消息、病例设定、情境等）
    if "guide_completed" not in st.session_state:
        st.session_state.guide_completed = False  # 当前引导是否完成（用于显示返回按钮）
    if "session_started" not in st.session_state:
        st.session_state.session_started = False  # 患者对话是否已开始
    if "show_guide_reminder" not in st.session_state:
        st.session_state.show_guide_reminder = False  # 是否显示“请找医小星”的提醒
    if "bg_guide_done" not in st.session_state:
        st.session_state.bg_guide_done = False  # 当前情境下背景教学是否已完成（点击后隐藏背景教学按钮）
    if "symptom_guide_done" not in st.session_state:
        st.session_state.symptom_guide_done = False  # 第一次症状分析引导是否已完成（医小星返回 [DONE] 后为 True）
    if "guide_in_progress" not in st.session_state:
        st.session_state.guide_in_progress = False  # 是否正处于引导模式（主要由 guide_mode 管理，用于临时控制）
    if "last_guided_message" not in st.session_state:
        st.session_state.last_guided_message = ""  # 上次引导对应的患者消息
    if "scene_narrator" not in st.session_state:
        st.session_state.scene_narrator = pd.DataFrame()

    # 通用智能问答（悬浮球入口）
    if "general_chat_mode" not in st.session_state:
        st.session_state.general_chat_mode = False
    if "general_messages" not in st.session_state:
        st.session_state.general_messages = []
    if "general_chat_prev" not in st.session_state:
        # 记录进入通用问答前的页面状态，便于返回
        st.session_state.general_chat_prev = {}


def go_home():
    """跳转到首页，并清空与系统/疾病相关的状态。"""
    st.session_state.current_page = "home"
    st.session_state.current_system = None
    st.session_state.current_disease = None
    st.session_state.situation_state = {}
    st.rerun()


def go_diseases():
    """跳转到疾病列表页，并清空与疾病相关的学习状态。"""
    st.session_state.current_page = "diseases"
    st.session_state.current_disease = None
    st.session_state.situation_state = {}
    st.rerun()


def go_learn():
    """跳转到学习页。"""
    st.session_state.current_page = "learn"
    st.rerun()


@st.cache_data
def _load_image_base64(absolute_path: str) -> str:
    """
    读取图片文件并返回 Base64 数据 URL（data:image/png;base64,...）。
    使用 st.cache_data 缓存，避免每次渲染重复读文件。路径需为绝对路径以保证缓存键稳定。
    若文件不存在或读取失败则返回空字符串。
    """
    if not absolute_path or not os.path.exists(absolute_path):
        return ""
    try:
        with open(absolute_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return f"data:image/png;base64,{b64}"
    except Exception:
        return ""


def _system_emoji(system_name: str, base_dir: str | None = None) -> str:
    """
    根据系统名称返回首页卡片图标：消化系统、泌尿系统为 Base64 内嵌的自定义图片 HTML，其余为 emoji。
    图片路径基于 __file__ 所在目录构建，确保任意工作目录下都能找到 data/images/ 中的文件。
    """
    name = str(system_name).strip()
    # 基于脚本所在目录解析 data/images/ 的绝对路径，不受当前工作目录影响
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 消化系统：gan.png → Base64 URL（缓存），失败则回退 emoji
    if name == "消化系统":
        path_gan = os.path.join(script_dir, "data", "images", "gan.png")
        src = _load_image_base64(path_gan)
        if src:
            return f'<img src="{src}" class="pbl-custom-icon" alt="消化系统">'
        return "🫃"
    # 泌尿系统：shen.png → Base64 URL（缓存），失败则回退 emoji
    if name == "泌尿系统":
        path_shen = os.path.join(script_dir, "data", "images", "shen.png")
        src = _load_image_base64(path_shen)
        if src:
            return f'<img src="{src}" class="pbl-custom-icon" alt="泌尿系统">'
        return "💧"
    # 其他系统：emoji
    mapping = {
        "循环系统": "🫀",
        "呼吸系统": "🫁",
        "神经系统": "🧠",
        "神经精神": "🧠",
        "消化系统": "🫃",
        "泌尿系统": "💧",
        "内分泌系统": "🧬",
        "血液系统": "🩸",
        "风湿免疫": "🦴",
        "骨科": "🦴",
        "感染": "🦠",
    }
    return mapping.get(name, "🩺")


# 内页系统背景图：系统名 → data/backgrounds/ 下文件名（不含扩展名，.jpg）
SYSTEM_BG_MAP = {
    "内分泌系统": "neifenmi",
    "呼吸系统": "huxi",
    "循环系统": "xunhuan",
    "泌尿系统": "miniao",
    "消化系统": "xiaohua",
    "神经精神": "shenjingjingshen",
    "骨科": "guke",
}

# ==================== 医小星图片配置 ====================
# [修改] 使用医小星图片 yixiaoxingemoji3.png（data/images/yixiaoxingemoji3.png）
MEDSTAR_ICON_FILENAME = "yixiaoxingemoji3.png"  # 图片文件名（必须放在 data/images/ 下）
MEDSTAR_ICON_SIZE = 50  # 图标显示尺寸（像素）
# =======================================================


def _inject_page_background(base_dir: str, page: str, system: str | None) -> None:
    """
    [修改] 根据当前页面与系统注入背景样式，同时强制内页所有普通文字为白色，按钮等交互元素保持深色。
    首页用 data/background.jpg；问问医小星 / 医小星深度精讲用 data/background1.1.jpg；
    其它内页用 data/backgrounds/{系统拼音}.jpg 或 background2/。
    回退顺序：系统背景图 → 首页背景图 → 渐变。
    """
    data_dir = os.path.join(base_dir, "data")
    home_bg = os.path.join(data_dir, "background.jpg")
    gradient_css = "background: linear-gradient(135deg, #e8f4f8 0%, #f0f4f8 50%, #f5f0e8 100%);"
    base_bg_css = "background-size: cover; background-repeat: no-repeat; background-position: center; background-attachment: fixed;"

    def try_load_b64(path: str) -> str:
        if not path or not os.path.exists(path):
            return ""
        try:
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
            return f"background-image: url(data:{mime};base64,{b64}); {base_bg_css}"
        except Exception:
            return ""

    medstar_feature_bg = os.path.join(data_dir, "background1.1.jpg")

    if page == "home":
        bg_css = try_load_b64(home_bg) or gradient_css
    elif page in ("general_chat", "case_list", "case_tutorial"):
        # 问问医小星（通用问答）、医小星深度精讲列表/详情：统一使用专用背景图
        bg_css = try_load_b64(medstar_feature_bg) or try_load_b64(home_bg) or gradient_css
    else:
        # [修改] 内页：按页面类型选择背景图文件夹
        # - diseases：只从 data/backgrounds/ 加载
        # - learn/simulation/report：优先 data/background2/，不存在则回退 data/backgrounds/
        name = (system or "").strip()
        stem = SYSTEM_BG_MAP.get(name, "")

        if page == "diseases":
            jpg_path = os.path.join(data_dir, "backgrounds", f"{stem}.jpg") if stem else ""
            png_path = os.path.join(data_dir, "backgrounds", f"{stem}.png") if stem else ""
            bg_css = (
                try_load_b64(jpg_path)
                or try_load_b64(png_path)
                or try_load_b64(home_bg)
                or gradient_css
            )
        elif page in ("learn", "simulation", "report"):
            jpg2_path = os.path.join(data_dir, "background2", f"{stem}.jpg") if stem else ""
            png2_path = os.path.join(data_dir, "background2", f"{stem}.png") if stem else ""
            jpg1_path = os.path.join(data_dir, "backgrounds", f"{stem}.jpg") if stem else ""
            png1_path = os.path.join(data_dir, "backgrounds", f"{stem}.png") if stem else ""
            bg_css = (
                try_load_b64(jpg2_path)
                or try_load_b64(png2_path)
                or try_load_b64(jpg1_path)
                or try_load_b64(png1_path)
                or try_load_b64(home_bg)
                or gradient_css
            )
        else:
            bg_css = try_load_b64(home_bg) or gradient_css

    st.markdown(
        f"""
        <style>
        .stApp {{
            font-family: 'PingFang SC', 'Microsoft YaHei', 'Comic Sans MS', 'Chalkboard SE', sans-serif;
            {bg_css}
        }}
        main .block-container {{
            background-color: rgba(255, 255, 255, 0.85);
            border-radius: 20px;
            padding: 2rem 3rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            margin-top: 18vh;
            margin-bottom: 5vh;
            max-width: 100%;
        }}
        /* [修改] 强制内页 block-container 及其所有子元素文字为白色 */
        main .block-container,
        main .block-container * {{
            color: white !important;
        }}
        /* [修改] 按钮、输入框、标签等交互元素恢复深色 */
        main .block-container button,
        main .block-container .stButton button,
        main .block-container input,
        main .block-container textarea,
        main .block-container .stTextInput input,
        main .block-container .stTextArea textarea,
        main .block-container [role="button"],
        main .block-container [type="button"],
        main .block-container [type="submit"],
        main .block-container label,
        main .block-container .stRadio label,
        main .block-container .stSelectbox label,
        main .block-container .stCheckbox label,
        main .block-container .stMultiSelect label,
        main .block-container .stSlider label,
        main .block-container .stDateInput label,
        main .block-container .stTimeInput label,
        main .block-container .stFileUploader label,
        main .block-container .stColorPicker label,
        main .block-container .stRadio span,
        main .block-container .stRadio div,
        main .block-container .stSelectbox span,
        main .block-container .stSelectbox div,
        main .block-container .stExpander,
        main .block-container .stAlert {{
            color: #2c3e50 !important;
        }}
        /* 标题样式保持原有设计 */
        h1, .pbl-home-title {{
            font-family: 'PingFang SC', 'Microsoft YaHei', 'Comic Sans MS', cursive;
            font-size: 3rem !important;
            font-weight: 700 !important;
            letter-spacing: 1px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }}
        .pbl-home-title {{ font-size: 2rem !important; }}
        h3 {{ font-size: 1.5rem !important; font-weight: 400 !important; margin-bottom: 2rem !important; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_floating_icon(base_dir: str) -> None:
    """
    [修改] 右下角悬浮元素（仅首页）：左侧医小星图片 + 右侧「和我聊聊天吧」按钮。
    使用紧凑列方案：固定定位 + width: fit-content + 列宽自适应 + 去除列间距 + 背景贴合内容。
    """
    # 只在首页显示，且在非引导/非通用问答模式
    if st.session_state.get("current_page") != "home":
        return
    if st.session_state.get("general_chat_mode", False) or st.session_state.get(
        "guide_mode", False
    ):
        return

    # 加载医小星图片（若失败则回退 emoji）
    img_path = os.path.join(base_dir, "data", "images", MEDSTAR_ICON_FILENAME)
    img_base64 = _load_image_base64(img_path)

    placeholder = st.empty()
    with placeholder.container():
        st.markdown(
            f"""
            <style>
            .compact-floating-container {{
                position: fixed;
                bottom: 20px;
                right: 20px;
                z-index: 99999;
                background-color: #4f8bf9;
                border-radius: 40px;
                padding: 4px 6px 4px 4px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                width: fit-content;
                display: flex;
                align-items: center;
                gap: 0;
            }}
            .compact-floating-container div[data-testid="column"] {{
                padding: 0 !important;
                margin: 0 !important;
                flex: none !important;
                width: auto !important;
            }}
            .compact-floating-container img.floating-medstar-img {{
                width: {MEDSTAR_ICON_SIZE}px;
                height: {MEDSTAR_ICON_SIZE}px;
                border-radius: 50%;
                display: block;
                box-shadow: 0 2px 8px rgba(0,0,0,0.2);
                transform: rotateY(15deg) rotateX(5deg);
                animation: float-medstar 3s ease-in-out infinite;
                transition: transform 0.3s;
            }}
            .compact-floating-container img.floating-medstar-img:hover {{
                transform: rotateY(15deg) rotateX(5deg) scale(1.1);
            }}
            .compact-floating-container .stButton button {{
                background: transparent !important;
                color: white !important;
                border: none !important;
                padding: 0 8px 0 2px !important;
                font-weight: 600 !important;
                font-size: 1.1rem !important;
                white-space: nowrap !important;
                box-shadow: none !important;
                margin: 0 !important;
                height: {MEDSTAR_ICON_SIZE}px !important;
                line-height: {MEDSTAR_ICON_SIZE}px !important;
                cursor: pointer !important;
            }}
            .compact-floating-container .stButton button:hover {{
                background: transparent !important;
                color: #f0f0f0 !important;
            }}
            @keyframes float-medstar {{
                0% {{ transform: rotateY(15deg) rotateX(5deg) translateY(0px); }}
                50% {{ transform: rotateY(15deg) rotateX(5deg) translateY(-4px); }}
                100% {{ transform: rotateY(15deg) rotateX(5deg) translateY(0px); }}
            }}
            </style>
            <div class="compact-floating-container">
            """,
            unsafe_allow_html=True,
        )

        cols = st.columns([1, 1])
        with cols[0]:
            if img_base64:
                st.markdown(
                    f'<img src="{img_base64}" class="floating-medstar-img">',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<span style="font-size: 40px; line-height: 50px;">🧑‍🏫</span>',
                    unsafe_allow_html=True,
                )

        with cols[1]:
            # 不使用 use_container_width，让按钮按内容宽度自适应，避免容器拉长
            if st.button("和我聊聊天吧", key="general_chat_compact"):
                st.session_state.general_chat_mode = True
                st.session_state.previous_page = st.session_state.get("current_page", "home")
                if st.session_state.get("current_system"):
                    st.session_state.previous_system = st.session_state.get("current_system")
                if st.session_state.get("current_disease"):
                    st.session_state.previous_disease = st.session_state.get("current_disease")
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


def _inject_home_global_styles(base_dir: str) -> None:
    """
    注入全局样式：背景图、内容区半透明、字体。
    base_dir 应为脚本所在目录（main 中为 os.path.dirname(os.path.abspath(__file__))），
    背景图路径为 base_dir/data/background.jpg；优先 Base64 嵌入，失败则回退渐变。
    背景图显示方式：cover 铺满、不重复、居中。保留作回退/辅助，main 中不再直接调用。
    """
    bg_path = os.path.join(base_dir, "data", "background.jpg")
    bg_css = "background-size: cover; background-repeat: no-repeat; background-position: center; background-attachment: fixed;"
    if os.path.exists(bg_path):
        try:
            with open(bg_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            bg_css = f"background-image: url(data:image/jpeg;base64,{b64}); background-size: cover; background-repeat: no-repeat; background-position: center; background-attachment: fixed;"
        except Exception:
            bg_css = "background: linear-gradient(135deg, #e8f4f8 0%, #f0f4f8 50%, #f5f0e8 100%);"
    else:
        bg_css = "background: linear-gradient(135deg, #e8f4f8 0%, #f0f4f8 50%, #f5f0e8 100%);"
    st.markdown(
        f"""
        <style>
        /* 全局字体与背景：圆润可爱风格，中英文回退 */
        .stApp {{
            font-family: 'PingFang SC', 'Microsoft YaHei', 'Comic Sans MS', 'Chalkboard SE', sans-serif;
            {bg_css}
        }}
        /* 内容区域：半透明白、圆角、阴影；margin-top 调大使首页卡片整体下移，避免遮挡背景 */
        main .block-container {{
            background-color: rgba(255, 255, 255, 0.85);
            border-radius: 20px;
            padding: 2rem 3rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            margin-top: 18vh;
            margin-bottom: 5vh;
            max-width: 100%;
        }}
        /* 标题：加大、圆润字体、字间距与柔和阴影（平台名统一为「临床启明星」） */
        h1, .pbl-home-title {{
            font-family: 'PingFang SC', 'Microsoft YaHei', 'Comic Sans MS', cursive;
            font-size: 3rem !important;
            font-weight: 700 !important;
            letter-spacing: 1px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }}
        .pbl-home-title {{ font-size: 2rem !important; }}
        h3 {{ font-size: 1.5rem !important; font-weight: 400 !important; margin-bottom: 2rem !important; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _inject_home_card_styles() -> None:
    """注入首页标题样式、网格卡片链接样式及自定义图标样式（.pbl-custom-icon 与 emoji 一致）。"""
    st.markdown(
        """
        <style>
        .pbl-home-title { font-size: 2rem; font-weight: 700; color: white; margin-bottom: 0.25rem; }
        .pbl-home-subtitle { font-size: 1.1rem; color: white; margin-bottom: 1.5rem; }
        /* 首页网格卡片链接：白底、圆角、阴影、悬停放大（与原按钮风格一致） */
        .pbl-home-card {
            display: flex; flex-direction: column; align-items: center; justify-content: center;
            background: rgba(255,255,255,0.92);
            color: #2c3e50;
            font-weight: 700;
            font-family: 'PingFang SC', 'Microsoft YaHei', sans-serif;
            border-radius: 16px;
            padding: 1.5rem 0.5rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            text-decoration: none;
            min-height: 100px;
            line-height: 1.4;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .pbl-home-card:hover { transform: scale(1.02); box-shadow: 0 6px 20px rgba(0,0,0,0.15); }
        .pbl-home-card-icon { font-size: 2rem; display: block; margin-bottom: 4px; text-align: center; }
        /* 消化/泌尿系统 Base64 自定义图片：与 emoji 大小一致、居中、圆角（原 _build_center_ring_html 内联样式同款） */
        .pbl-custom-icon {
            width: 2rem;
            height: 2rem;
            object-fit: contain;
            display: block;
            margin: 0 auto;
            border-radius: 50%;
        }
        /* 首页系统卡片按钮样式（使用 Streamlit 按钮实现） */
        .home-system-button .stButton button {
            background: rgba(255,255,255,0.92) !important;
            color: #2c3e50 !important;
            border-radius: 16px !important;
            padding: 1.5rem 0.5rem !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
            border: none !important;
            font-weight: 700 !important;
            font-size: 1.2rem !important;
            line-height: 1.4 !important;
            height: auto !important;
            white-space: normal !important;
            transition: transform 0.2s, box-shadow 0.2s !important;
        }
        .home-system-button .stButton button:hover {
            transform: scale(1.02) !important;
            box-shadow: 0 6px 20px rgba(0,0,0,0.15) !important;
        }
        /* 移动端触摸优化：增大点击区域，去掉高亮 */
        .home-system-button .stButton button {
            min-height: 100px !important;
            touch-action: manipulation !important;
            -webkit-tap-highlight-color: transparent !important;
        }
        .home-system-button .stButton button:active {
            opacity: 0.8 !important;
            transform: scale(0.98) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _home_card_html(system_name: str, base_dir: str | None) -> str:
    """生成首页单张卡片的 HTML（<a> 内为图标 + 系统名），图标可为 emoji 或自定义 img。"""
    icon = _system_emoji(system_name, base_dir)
    q = quote(system_name)
    return (
        f'<a href="?system={q}" class="pbl-home-card" '
        f'style="-webkit-tap-highlight-color: transparent; touch-action: manipulation;">'
        f'<span class="pbl-home-card-icon">{icon}</span>'
        f'<span class="pbl-home-card-name">{system_name}</span></a>'
    )


def render_home(df_mcq: pd.DataFrame, base_dir: str | None = None) -> None:
    """
    首页：网格布局——第一行四个系统卡片、第二行三个居中。
    卡片为 HTML 链接（支持消化/泌尿系统自定义图片），点击通过 ?system= 跳转，main() 中 query_params 写入 current_system。
    """
    _inject_home_card_styles()

    # [修改] 恢复首页标题原样
    st.markdown('<p class="pbl-home-title">⭐ 临床启明星</p>', unsafe_allow_html=True)

    st.markdown('<p class="pbl-home-subtitle">请选择系统开始学习</p>', unsafe_allow_html=True)

    # 硬编码七个系统，保证布局稳定（不再从题库动态获取）
    systems = [
        "内分泌系统", "呼吸系统", "循环系统", "泌尿系统",
        "消化系统", "神经精神", "骨科",
    ]

    # 第一行：st.columns(4)，四个系统按钮（使用原生按钮，移动端点击更可靠）
    cols_top = st.columns(4)
    for i, sys in enumerate(systems[:4]):
        with cols_top[i]:
            with st.container():
                st.markdown('<div class="home-system-button">', unsafe_allow_html=True)
                label = f"{_system_emoji(sys, base_dir)}\n\n{sys}"
                if st.button(label, key=f"home_sys_{sys}", use_container_width=True):
                    st.session_state.current_system = sys
                    st.session_state.current_page = "diseases"
                    st.session_state.current_disease = None
                    st.session_state.situation_state = {}
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

    # 第二行：中间三列系统按钮，左右留白使其居中
    col_left, col_mid1, col_mid2, col_mid3, col_right = st.columns([1, 1, 1, 1, 1])
    with col_mid1:
        sys = systems[4]
        with st.container():
            st.markdown('<div class="home-system-button">', unsafe_allow_html=True)
            label = f"{_system_emoji(sys, base_dir)}\n\n{sys}"
            if st.button(label, key=f"home_sys_{sys}", use_container_width=True):
                st.session_state.current_system = sys
                st.session_state.current_page = "diseases"
                st.session_state.current_disease = None
                st.session_state.situation_state = {}
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
    with col_mid2:
        sys = systems[5]
        with st.container():
            st.markdown('<div class="home-system-button">', unsafe_allow_html=True)
            label = f"{_system_emoji(sys, base_dir)}\n\n{sys}"
            if st.button(label, key=f"home_sys_{sys}", use_container_width=True):
                st.session_state.current_system = sys
                st.session_state.current_page = "diseases"
                st.session_state.current_disease = None
                st.session_state.situation_state = {}
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
    with col_mid3:
        sys = systems[6]
        with st.container():
            st.markdown('<div class="home-system-button">', unsafe_allow_html=True)
            label = f"{_system_emoji(sys, base_dir)}\n\n{sys}"
            if st.button(label, key=f"home_sys_{sys}", use_container_width=True):
                st.session_state.current_system = sys
                st.session_state.current_page = "diseases"
                st.session_state.current_disease = None
                st.session_state.situation_state = {}
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    # 首页右下角固定按钮（marker + 相邻兄弟选择器；顺序：上→下 问问医小星 / 深度精讲 / 联系我们）
    st.markdown(
        """
        <style>
        .home-fixed-buttons-marker { display: none !important; }
        .home-fixed-buttons-marker + div[data-testid="stButton"],
        .home-fixed-buttons-marker + div[data-testid="stButton"] + div[data-testid="stButton"],
        .home-fixed-buttons-marker + div[data-testid="stButton"] + div[data-testid="stButton"] + div[data-testid="stButton"] {
            position: fixed !important;
            right: 30px !important;
            z-index: 9999 !important;
            margin: 0 !important;
        }
        .home-fixed-buttons-marker + div[data-testid="stButton"] { bottom: 130px !important; }
        .home-fixed-buttons-marker + div[data-testid="stButton"] + div[data-testid="stButton"] {
            bottom: 80px !important;
        }
        .home-fixed-buttons-marker + div[data-testid="stButton"] + div[data-testid="stButton"] + div[data-testid="stButton"] {
            bottom: 30px !important;
        }
        .home-fixed-buttons-marker + div[data-testid="stButton"] button,
        .home-fixed-buttons-marker + div[data-testid="stButton"] + div[data-testid="stButton"] button,
        .home-fixed-buttons-marker + div[data-testid="stButton"] + div[data-testid="stButton"] + div[data-testid="stButton"] button {
            background-color: #0A2F6C !important;
            color: white !important;
            font-size: 1.2rem !important;
            padding: 8px 16px !important;
            border-radius: 8px !important;
            border: none !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2) !important;
            transition: background-color 0.2s !important;
            white-space: nowrap !important;
            width: auto !important;
        }
        .home-fixed-buttons-marker + div[data-testid="stButton"] button:hover,
        .home-fixed-buttons-marker + div[data-testid="stButton"] + div[data-testid="stButton"] button:hover,
        .home-fixed-buttons-marker + div[data-testid="stButton"] + div[data-testid="stButton"] + div[data-testid="stButton"] button:hover {
            background-color: #1E4A8C !important;
        }
        </style>
        <div class="home-fixed-buttons-marker"></div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("⭐ 问问医小星", key="ask_medstar_button"):
        st.session_state.general_chat_mode = True
        st.session_state.previous_page = st.session_state.current_page
        st.rerun()
    if st.button("✨ 医小星深度精讲", key="case_list_button"):
        st.session_state.current_page = "case_list"
        st.rerun()
    # 第三个按钮：联系我们（跳转独立页面）
    if st.button("📧 联系我们", key="contact_us_button"):
        st.session_state.current_page = "contact"
        st.rerun()


def inner_page_style() -> str:
    """返回内页（疾病/学习/报告/模拟问诊）使用的深色背景+白字 CSS，首页不调用以保持原样。"""
    return """
<style>
    /* 移动端触摸优化：全局 touch-action，减少点击延迟 */
    * {
        touch-action: manipulation;
    }
    /* 内页全局文字白色 */
    .stApp, .stMarkdown, p, h1, h2, h3, h4, h5, h6, div, span, label, .stChatMessage, .stSelectbox, .stRadio, .stButton button {
        color: #ffffff !important;
    }
    /* 放大内页整体文字 */
    main .block-container,
    main .block-container * {
        font-size: 1.2rem !important;
    }
    /* 背景色 */
    .stApp {
        background-color: #1e1e1e;
    }
    /* 输入框、文本区域背景调暗；选择框主体单独在后面覆盖为白底黑字 */
    input, textarea, .stTextInput input, .stNumberInput input {
        background-color: #2d2d2d !important;
        color: white !important;
        border-color: #444 !important;
        font-size: 1.2rem !important;
    }
    /* 全局下拉框主体：白底黑字（解决白底白字问题） */
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: white !important;
        color: black !important;
        border-color: #cccccc !important;
    }
    .stSelectbox div[data-baseweb="select"] span {
        color: black !important;
    }
    /* 按钮背景色（可自定义） */
    .stButton button {
        background-color: #0066cc !important;
        border: none;
    }
    .stButton button:hover {
        background-color: #004999 !important;
    }
    /* 内页标题字号调整，更醒目 */
    h1 {
        font-size: 2.5rem !important;
    }
    h2 {
        font-size: 2rem !important;
    }
    h3 {
        font-size: 1.8rem !important;
    }
    /* 卡片、展开器等保持半透明背景 */
    .stExpander, .stAlert, .stInfo, .stSuccess, .stWarning, .stError {
        background-color: rgba(255,255,255,0.1) !important;
    }
    /* 放大疾病名称（加粗 strong 一般用于疾病名） */
    strong {
        font-size: 1.2rem !important;
    }
    /* 疾病列表列内 Markdown 与按钮之间的间距更紧凑 */
    div[data-testid="column"] .stMarkdown {
        margin-bottom: 0.3rem !important;
    }
    /* 提高按钮可点击区域（移动端优化） */
    .stButton button {
        min-height: 44px;
        min-width: 44px;
        padding: 10px 15px;
        font-size: 16px !important;
        cursor: pointer;
        -webkit-tap-highlight-color: transparent;
    }
    /* 触摸反馈 */
    .stButton button:active {
        opacity: 0.7;
        transform: scale(0.98);
    }
    /* 确保按钮容器不被遮挡 */
    .stButton {
        z-index: 10;
        position: relative;
    }
    /* 全局下拉列表选项：统一黑字白底，避免白底白字 */
    ul[role="listbox"] {
        background-color: #ffffff !important;
    }
    ul[role="listbox"] li,
    ul[role="listbox"] li * {
        color: #000000 !important;
    }
    /* 禁用任何可能遮挡点击的父容器（若有自定义卡片） */
    .stApp [data-testid="stVerticalBlock"] {
        pointer-events: auto !important;
    }
    /* 针对首页系统卡片（如果用了自定义 div），确保按钮可点 */
    .system-card .stButton button {
        width: 100%;
        margin-top: 5px;
    }
</style>
"""


def render_diseases(df_mcq: pd.DataFrame) -> None:
    """疾病列表页：根据当前系统展示疾病列表。内页标题与正文使用内联白字。"""
    st.markdown(inner_page_style(), unsafe_allow_html=True)
    st.markdown('<h1 style="color: white !important;">临床启明星</h1>', unsafe_allow_html=True)

    if st.button("返回首页"):
        go_home()

    current_system = st.session_state.current_system
    if not current_system:
        st.info("当前未选择系统，请先在首页选择系统。")
        return

    st.markdown(f'<h2 style="color: white !important;">当前系统：{current_system}</h2>', unsafe_allow_html=True)
    df_sys = df_mcq[df_mcq["系统"].astype(str) == str(current_system)]

    diseases = (
        df_sys["疾病"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    if not diseases:
        st.warning("该系统下未找到任何疾病。")
        return

    st.markdown('<h3 style="color: white !important;">请选择疾病</h3>', unsafe_allow_html=True)
    cols_per_row = 2  # 每行显示两个疾病块
    for i, disease in enumerate(diseases):
        if i % cols_per_row == 0:
            cols = st.columns(cols_per_row)
        with cols[i % cols_per_row]:
            st.markdown(f'<p style="color: white !important;"><strong>{disease}</strong></p>', unsafe_allow_html=True)
            col1, col2 = st.columns(2, gap="small")
            with col1:
                if st.button("开始学习", key=f"learn_{disease}", use_container_width=True):
                    # 调试：确认点击被捕获
                    st.write(f"点击了开始学习: {disease}")
                    st.session_state.current_disease = disease
                    st.session_state.situation_state = {}
                    go_learn()
            with col2:
                if st.button("模拟问诊", key=f"sim_{disease}", use_container_width=True):
                    # 调试：确认点击被捕获
                    st.write(f"点击了模拟问诊: {disease}")
                    st.session_state.current_disease = disease
                    st.session_state.current_page = "simulation"
                    st.rerun()


def get_or_init_situation_state(situation_key: Any) -> Dict[str, Any]:
    """
    获取或初始化某个情境的状态。
    返回的结构示例：
        {
            "q_index": int,
            "answers": {idx: "A"/"B"/...},
            "submitted": set(),
            "correct": {idx: bool},
            "knowledge": {idx: str},
            "essay_answer": str,
            "essay_submitted": bool,
            "essay_score": float,
            "essay_max": float,
            "essay_missing": list[str],
            "has_essay": bool,
        }
    """
    if situation_key not in st.session_state.situation_state:
        st.session_state.situation_state[situation_key] = {
            "q_index": 0,
            "answers": {},
            "submitted": set(),
            "correct": {},
            "knowledge": {},
            "essay_answer": "",
            "essay_submitted": False,
            "essay_score": 0.0,
            "essay_max": 0.0,
            "essay_missing": [],
            "has_essay": False,
        }
    else:
        # 兼容旧会话中缺失字段的情况
        state = st.session_state.situation_state[situation_key]
        state.setdefault("answers", {})
        state.setdefault("submitted", set())
        state.setdefault("correct", {})
        state.setdefault("knowledge", {})
        state.setdefault("essay_answer", "")
        state.setdefault("essay_submitted", False)
        state.setdefault("essay_score", 0.0)
        state.setdefault("essay_max", 0.0)
        state.setdefault("essay_missing", [])
        state.setdefault("has_essay", False)
    return st.session_state.situation_state[situation_key]


def render_situation_tab(
    situation_value: Any,
    df_sit: pd.DataFrame,
    df_short_q: pd.DataFrame,
    df_scoring: pd.DataFrame,
    disease: str,
    grade: str,
) -> None:
    """
    在学习页中渲染单个情境标签页的内容（选择题 + 简答题）。
    df_sit 已经过滤为当前疾病、当前情境、年级为指定年级的题目，并按“阶”升序排列。
    """
    state = get_or_init_situation_state(situation_value)

    # 显示案例描述（若存在）
    if not df_sit.empty:
        case_desc = df_sit.iloc[0].get("案例描述", "")
        if pd.notna(case_desc) and str(case_desc).strip():
            with st.expander("📋 案例描述", expanded=True):
                st.markdown(f'<span style="color: white !important;">{str(case_desc)}</span>', unsafe_allow_html=True)

    # 选择题区域
    st.markdown('<h3 style="color: white !important;">选择题</h3>', unsafe_allow_html=True)

    if df_sit.empty:
        st.info("该情境下暂无选择题。")
    else:
        num_questions = len(df_sit)
        q_index = state["q_index"]

        # 防止索引越界
        if q_index < 0:
            q_index = 0
        if q_index >= num_questions:
            q_index = num_questions - 1
        state["q_index"] = q_index

        row = df_sit.iloc[q_index]
        explanation = row.get("解析", "")

        st.markdown(f'<p style="color: white !important;"><strong>当前题目：第 {q_index + 1} / {num_questions} 题</strong></p>', unsafe_allow_html=True)
        st.markdown(f'<p style="color: white !important;"><strong>阶：</strong> {row.get("阶", "")}</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="color: white !important;"><strong>问题：</strong> {row.get("问题", "")}</p>', unsafe_allow_html=True)

        options = {
            "A": row.get("选项A", ""),
            "B": row.get("选项B", ""),
            "C": row.get("选项C", ""),
            "D": row.get("选项D", ""),
        }

        # 去除空选项
        options = {
            k: v for k, v in options.items() if pd.notna(v) and str(v).strip() != ""
        }

        if not options:
            st.warning("该题目没有可用的选项。")
        else:
            # 准备选项字典：键为选项字母，值为显示文本
            options_dict = {k: v for k, v in options.items()}

            # 当前题目已经保存的答案（默认选第一个选项）
            option_keys = list(options_dict.keys())
            current_answer = state["answers"].get(
                q_index,
                option_keys[0] if option_keys else None,
            )

            answer_key = f"radio_{situation_value}_{q_index}"

            if current_answer is not None and current_answer in option_keys:
                default_index = option_keys.index(current_answer)
            else:
                default_index = 0

            # Radio 返回值为选项字母本身，显示文本通过 format_func 格式化
            selected_option = st.radio(
                "请选择一个选项：",
                options=option_keys,
                index=default_index,
                format_func=lambda x: f"{x}. {options_dict[x]}",
                key=answer_key,
            )

            # 保存当前题目的选择
            if selected_option is not None:
                state["answers"][q_index] = selected_option

            col_prev, col_submit, col_next = st.columns(3)

            with col_prev:
                if st.button(
                    "上一题",
                    disabled=q_index == 0,
                    key=f"prev_{situation_value}_{q_index}",
                ):
                    state["q_index"] = max(0, q_index - 1)
                    st.rerun()

            with col_submit:
                if st.button(
                    "提交本题",
                    key=f"submit_q_{situation_value}_{q_index}",
                ):
                    user_choice = state["answers"].get(q_index)
                    correct_answer = str(row.get("答案", "")).strip().upper()
                    # 记录该题的知识点，供报告统计
                    knowledge_point = row.get("知识点", "")
                    state["knowledge"][q_index] = knowledge_point

                    if not user_choice:
                        st.warning("请先选择一个选项。")
                    else:
                        # 只支持单字母答案
                        if len(correct_answer) == 1 and correct_answer in [
                            "A",
                            "B",
                            "C",
                            "D",
                        ]:
                            is_correct = user_choice == correct_answer
                            state["submitted"].add(q_index)
                            state["correct"][q_index] = is_correct
                            if is_correct:
                                st.success("回答正确！")
                            else:
                                st.error("回答错误。")
                                st.info(f"正确答案为：{correct_answer}")
                        else:
                            state["submitted"].add(q_index)
                            state["correct"][q_index] = False
                            st.warning(
                                "该题为多选题或答案格式不支持，暂不支持自动判分。"
                            )
                            if correct_answer:
                                st.info(f"参考答案：{correct_answer}")

            with col_next:
                if st.button(
                    "下一题",
                    disabled=q_index >= num_questions - 1,
                    key=f"next_{situation_value}_{q_index}",
                ):
                    state["q_index"] = min(num_questions - 1, q_index + 1)
                    st.rerun()

            # 已提交标记
            if q_index in state["submitted"]:
                is_correct = state["correct"].get(q_index, False)
                if is_correct:
                    st.markdown('<p style="color: white !important;">✅ 本题已提交：<strong>回答正确</strong></p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p style="color: white !important;">❌ 本题已提交：<strong>回答错误或未判分</strong></p>', unsafe_allow_html=True)
                if explanation and pd.notna(explanation):
                    st.markdown(
                        f'<p style="color: white !important; background-color: rgba(255,255,255,0.1); padding: 8px; border-radius: 4px; margin-top: 5px;"><strong>解析：</strong> {explanation}</p>',
                        unsafe_allow_html=True,
                    )

    st.markdown("---")

    # 简答题区域
    st.markdown('<h3 style="color: white !important;">简答题</h3>', unsafe_allow_html=True)

    # 检查简答题题库是否已正确加载
    if df_short_q is None or df_short_q.empty:
        state["has_essay"] = False
        st.info("简答题题库文件为空或未正确加载，请联系管理员检查数据文件。")
    else:
        # 统一按字符串比较疾病、情境和年级等级
        df_essay = df_short_q[
            (df_short_q["疾病"].astype(str) == str(disease))
            & (df_short_q["情境"].astype(str) == str(situation_value))
            & (df_short_q["年级等级"].astype(str) == str(grade))
        ]

        if df_essay.empty:
            state["has_essay"] = False
            st.info("本情境无简答题（或当前年级无对应题目）。")
        else:
            state["has_essay"] = True
            essay_row = df_essay.iloc[0]
            question_text = essay_row.get("简答题题目", "")
            st.markdown(f'<p style="color: white !important;"><strong>题目：</strong> {question_text}</p>', unsafe_allow_html=True)

            text_key = f"essay_{situation_value}"
            default_text = state.get("essay_answer", "")

            user_text = st.text_area(
                "请输入你的答案：",
                value=default_text,
                height=150,
                key=text_key,
            )
            state["essay_answer"] = user_text

            if st.button("提交简答题", key=f"submit_essay_{situation_value}"):
                score, max_score, missing = score_short_answer(
                    user_text, df_scoring, disease, grade, situation_value
                )
                state["essay_submitted"] = True
                state["essay_score"] = score
                state["essay_max"] = max_score
                state["essay_missing"] = missing

                if max_score > 0:
                    st.success(f"简答题得分：{score} / {max_score}")
                else:
                    st.info("当前简答题暂无可用评分规则。")

                if missing:
                    st.warning(
                        "你可能遗漏了以下关键词（仅供参考）："
                        + "、".join(str(m) for m in missing)
                    )

            if state.get("essay_submitted"):
                if state.get("essay_max", 0) > 0:
                    st.markdown(
                        f'<p style="color: white !important;">✅ 简答题已提交，得分：<strong>{state.get("essay_score", 0)} / {state.get("essay_max", 0)}</strong></p>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown('<p style="color: white !important;">✅ 简答题已提交。</p>', unsafe_allow_html=True)

                # 显示参考答案（若已加载参考答案表）
                ref_df = getattr(st.session_state, "short_answer_refs", pd.DataFrame())
                if isinstance(ref_df, pd.DataFrame) and not ref_df.empty:
                    mask = (
                        ref_df["疾病"].astype(str) == str(disease)
                    ) & (
                        ref_df["情境"].astype(str) == str(situation_value)
                    )
                    mask = mask & ref_df["年级等级"].astype(str).str.contains(
                        str(grade), na=False
                    )
                    matches = ref_df[mask]
                    if not matches.empty:
                        ref_text = matches.iloc[0].get("参考答案", "")
                        if pd.notna(ref_text) and str(ref_text).strip():
                            st.markdown(
                                f'<p style="color: white !important; background-color: rgba(255,255,255,0.1); padding: 8px; border-radius: 4px; margin-top: 5px;"><strong>📖 参考答案要点：</strong> {ref_text}</p>',
                                unsafe_allow_html=True,
                            )


def render_learn(
    df_mcq: pd.DataFrame,
    df_short_q: pd.DataFrame,
    df_scoring: pd.DataFrame,
) -> None:
    """学习页：按情境展示选择题与简答题，并支持年级切换。内页标题使用内联白字。"""
    st.markdown(inner_page_style(), unsafe_allow_html=True)
    st.markdown('<h1 style="color: white !important;">临床启明星</h1>', unsafe_allow_html=True)

    # 返回按钮
    col_back_home, col_back_disease = st.columns(2)
    with col_back_home:
        if st.button("返回首页", key="back_to_home_from_learn"):
            go_home()
    with col_back_disease:
        if st.button("返回疾病列表", key="back_to_disease_from_learn"):
            go_diseases()

    # 年级选择器（初级 / 高级）
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        grade_options = ["初级", "高级"]
        current_grade = st.session_state.get("grade", "初级")
        try:
            default_index = grade_options.index(current_grade)
        except ValueError:
            default_index = 0

        selected_grade = st.radio(
            "选择年级",
            grade_options,
            index=default_index,
            horizontal=True,
            key="grade_selector",
        )

        # 年级切换时重置情境状态并刷新页面
        if selected_grade != st.session_state.get("grade", "初级"):
            st.session_state.grade = selected_grade
            st.session_state.situation_state = {}
            st.rerun()

    current_system = st.session_state.current_system
    current_disease = st.session_state.current_disease

    if not current_system or not current_disease:
        st.info("当前未选择系统或疾病，请先返回重新选择。")
        return

    st.markdown(f'<h2 style="color: white !important;">当前系统：{current_system}</h2>', unsafe_allow_html=True)
    st.markdown(f'<h2 style="color: white !important;">当前疾病：{current_disease}</h2>', unsafe_allow_html=True)

    # 使用 session_state 中的年级信息
    grade = st.session_state.get("grade", "初级")
    st.session_state.grade = grade

    # 统一将筛选条件转换为字符串，避免类型不匹配
    df_filtered = df_mcq[
        (df_mcq["系统"].astype(str) == str(current_system))
        & (df_mcq["疾病"].astype(str) == str(current_disease))
        & (df_mcq["年级等级"].astype(str) == str(grade))
    ]

    if df_filtered.empty:
        st.warning(
            f"在当前系统“{current_system}”/疾病“{current_disease}”/年级“{grade}”下未找到任何题目。"
        )
        st.info("请检查题库中的“系统”、“疾病”、“年级等级”列是否与当前选择一致。")
        return

    # 将情境统一转换为字符串，避免 float/int 混用导致匹配问题
    situations = (
        df_filtered["情境"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    if not situations:
        st.warning(
            "当前疾病下未找到任何情境，请检查题库中的“情境”列是否已正确填写。"
        )
        return

    tabs = st.tabs([f"情境 {i + 1}" for i in range(len(situations))])

    for tab, sit in zip(tabs, situations):
        with tab:
            # 用字符串比较情境值，避免类型不一致
            df_sit = (
                df_filtered[df_filtered["情境"].astype(str) == str(sit)]
                .sort_values(by="阶")
                .reset_index(drop=True)
            )
            render_situation_tab(
                sit,
                df_sit,
                df_short_q,
                df_scoring,
                current_disease,
                grade,
            )

    # 完成检测：所有情境的选择题与简答题是否均已完成
    completion_info = {}
    all_done = True
    for sit in situations:
        # 使用与上面一致的情境筛选方式，统一按字符串比较
        df_sit_done = df_filtered[df_filtered["情境"].astype(str) == str(sit)]
        num_q = len(df_sit_done)
        state = get_or_init_situation_state(sit)

        submitted_cnt = len(state["submitted"])
        has_essay = state.get("has_essay", False)
        essay_submitted = state.get("essay_submitted", False)

        completion_info[str(sit)] = {
            "选择题已提交数量": submitted_cnt,
            "选择题总数": num_q,
            "有简答题": has_essay,
            "简答题已提交": essay_submitted if has_essay else None,
        }

        if num_q > 0 and submitted_cnt != num_q:
            all_done = False
            continue

        if has_essay and not essay_submitted:
            all_done = False
            continue

    if all_done:
        st.markdown("---")
        st.success("当前疾病所有情境的学习已完成。")  # 提示框保持默认样式
        if st.button("查看学习报告"):
            st.session_state.current_page = "report"
            st.rerun()


def score_short_answer(
    user_answer: str,
    df_scoring: pd.DataFrame,
    disease: str,
    grade: str,
    situation: Any = None,
) -> tuple[float, float, list[str]]:
    """
    对简答题答案进行评分。
    如果评分文件中有“情境”列，则按疾病+情境+年级筛选；
    否则按疾病+年级筛选（兼容旧版）。
    疾病名称会自动去除首尾的引号和空格，以便匹配。
    """
    if df_scoring is None or df_scoring.empty:
        return 0.0, 0.0, []

    def _clean_text(x: Any) -> str:
        """
        更强的文本标准化：
        - 去除首尾空白与单双引号
        - 去除常见不可见字符（BOM/零宽/软连字符等）
        """
        s = str(x) if x is not None else ""
        s = s.strip().strip('"').strip("'")
        for ch in ("\ufeff", "\u200b", "\u200c", "\u200d", "\u00ad"):
            s = s.replace(ch, "")
        return s.strip()

    def _norm_situation(x: Any) -> str:
        """将 1、1.0、'1 ' 等统一成 '1'，无法解析则退回清洗后的字符串。"""
        if x is None:
            return ""
        try:
            # 兼容 Excel 读取出的 float 情境（1.0）
            return str(int(float(str(x).strip())))
        except Exception:
            return _clean_text(x)

    # 标准化疾病名称：去除首尾的引号、空格、不可见字符
    disease_clean = _clean_text(disease)

    # 按疾病筛选（对评分表中的疾病列也做同样处理）
    df_scoring = df_scoring.copy()
    df_scoring["疾病_clean"] = df_scoring["疾病"].astype(str).apply(_clean_text)

    df_disease = df_scoring[df_scoring["疾病_clean"] == disease_clean]
    if df_disease.empty:
        return 0.0, 0.0, []

    # 再按适用年级筛选
    mask = df_disease["适用年级"].astype(str).str.contains(str(grade), na=False)
    df_grade = df_disease[mask]
    if df_grade.empty:
        return 0.0, 0.0, []

    # 如果评分表中有“情境”列且传入了情境，则进一步按情境筛选（兼容 1 vs 1.0，且支持多情境如 "1,2"）
    if "情境" in df_grade.columns and situation is not None:
        target_sit = _norm_situation(situation)

        def sit_contains(row_sit: Any) -> bool:
            if pd.isna(row_sit):
                return False
            parts = str(row_sit).split(",")
            normalized = {_norm_situation(p) for p in parts if str(p).strip() != ""}
            return target_sit in normalized

        df_grade = df_grade[df_grade["情境"].apply(sit_contains)]
        if df_grade.empty:
            return 0.0, 0.0, []

    total_score = 0.0
    max_score = 0.0
    missing: list[str] = []

    user_answer_norm = (user_answer or "").lower()

    for _, row in df_grade.iterrows():
        keyword = str(row.get("关键词", "")).strip().lower()
        if not keyword:
            continue

        score_val = row.get("分值", 0)
        if pd.isna(score_val):
            score_val = 0
        try:
            score_val = float(score_val)
        except (TypeError, ValueError):
            score_val = 0

        max_score += score_val

        synonyms = [keyword]
        for i in range(1, 5):
            syn = row.get(f"同义词{i}", "")
            if pd.notna(syn) and str(syn).strip():
                synonyms.append(str(syn).lower().strip())

        found = any(syn in user_answer_norm for syn in synonyms)
        if found:
            total_score += score_val
        else:
            missing.append(keyword)

    return total_score, max_score, missing


def render_report(
    df_mcq: pd.DataFrame,
    df_short_q: pd.DataFrame,
    df_scoring: pd.DataFrame,
) -> None:
    """
    学习报告页：展示选择题掌握情况、简答题表现及复习建议。
    依赖 session_state 中的 situation_state 数据。内页标题与正文使用内联白字。
    """
    st.markdown(inner_page_style(), unsafe_allow_html=True)
    st.markdown('<h1 style="color: white !important;">临床启明星 - 学习报告</h1>', unsafe_allow_html=True)

    # 顶部导航按钮
    col_back_learn, col_back_disease, col_back_home = st.columns(3)
    with col_back_learn:
        if st.button("返回学习页"):
            st.session_state.current_page = "learn"
            st.rerun()
    with col_back_disease:
        if st.button("返回疾病列表"):
            go_diseases()
    with col_back_home:
        if st.button("返回首页"):
            go_home()

    current_system = st.session_state.get("current_system")
    current_disease = st.session_state.get("current_disease")
    grade = st.session_state.get("grade", "初级")

    if not current_disease:
        st.info("当前未选择任何疾病，无法生成学习报告。请先返回并完成学习。")
        return

    if current_system:
        st.markdown(f'<h2 style="color: white !important;">当前系统：{current_system}</h2>', unsafe_allow_html=True)
    st.markdown(f'<h2 style="color: white !important;">当前疾病：{current_disease}</h2>', unsafe_allow_html=True)
    st.markdown(f'<p style="color: white !important;"><strong>年级：</strong> {grade}</p>', unsafe_allow_html=True)

    situation_state: Dict[Any, Dict[str, Any]] = st.session_state.get(
        "situation_state", {}
    )
    if not situation_state:
        st.info("当前疾病暂无任何学习记录。")
        return

    # ------------------------------
    # 1. 选择题掌握情况
    # ------------------------------
    st.markdown('<h2 style="color: white !important;">选择题掌握情况</h2>', unsafe_allow_html=True)

    knowledge_stats: Dict[str, Dict[str, int]] = {}

    for sit_key, state in situation_state.items():
        submitted = state.get("submitted", set())
        correct_dict = state.get("correct", {})
        knowledge_dict = state.get("knowledge", {})

        for q_idx in submitted:
            knowledge_point = str(knowledge_dict.get(q_idx, "")).strip()
            if not knowledge_point:
                # 没有知识点标注的题目暂时不计入统计
                continue

            if knowledge_point not in knowledge_stats:
                knowledge_stats[knowledge_point] = {"correct": 0, "total": 0}

            knowledge_stats[knowledge_point]["total"] += 1
            if correct_dict.get(q_idx, False):
                knowledge_stats[knowledge_point]["correct"] += 1

    weak_points: list[str] = []

    if not knowledge_stats:
        st.info("尚未记录到任何带有知识点标注的选择题作答数据。")
    else:
        with st.expander("查看各知识点正确率", expanded=True):
            for kp, stat in knowledge_stats.items():
                total = stat["total"]
                correct = stat["correct"]
                if total == 0:
                    continue
                acc = correct / total
                percent = acc * 100
                st.markdown(f'<span style="color: white !important;">{kp}：{correct}/{total} ({percent:.1f}%)</span>', unsafe_allow_html=True)
                st.progress(min(max(acc, 0.0), 1.0))
                if acc < 0.6:
                    weak_points.append(kp)

    # ------------------------------
    # 2. 简答题表现
    # ------------------------------
    st.markdown('<h2 style="color: white !important;">简答题表现</h2>', unsafe_allow_html=True)

    total_essay_score = 0.0
    total_essay_max = 0.0
    essay_rows = []
    all_missing_keywords: set[str] = set()

    for sit_key, state in situation_state.items():
        if not state.get("has_essay", False):
            continue
        if not state.get("essay_submitted", False):
            continue

        score = float(state.get("essay_score", 0.0) or 0.0)
        max_score = float(state.get("essay_max", 0.0) or 0.0)
        missing_list = state.get("essay_missing", []) or []

        total_essay_score += score
        total_essay_max += max_score

        for m in missing_list:
            if m:
                all_missing_keywords.add(str(m))

        essay_rows.append(
            {
                "情境": str(sit_key),
                "得分": score,
                "总分": max_score,
                "漏掉的关键词": "、".join(str(m) for m in missing_list if m),
            }
        )

    if total_essay_max > 0:
        st.metric(
            "简答题总得分",
            f"{total_essay_score:.1f} / {total_essay_max:.1f}",
        )
    else:
        st.info("尚未记录到任何已评分的简答题。")

    if essay_rows:
        df_essay_report = pd.DataFrame(essay_rows)
        st.table(df_essay_report)

    # ------------------------------
    # 3. 复习建议
    # ------------------------------
    st.markdown('<h2 style="color: white !important;">复习建议</h2>', unsafe_allow_html=True)

    if not weak_points and not all_missing_keywords:
        st.success("整体表现良好，暂未发现明显薄弱知识点或遗漏关键词。请继续保持。")
    else:
        if weak_points:
            st.info(
                "选择题薄弱知识点："
                + "、".join(str(kp) for kp in weak_points)
                + "。建议重点复习相关内容。"
            )
        if all_missing_keywords:
            st.warning(
                "你在简答题中可能遗漏了以下关键词："
                + "、".join(sorted(all_missing_keywords))
                + "。请结合教材或讲义，重点复习这些概念。"
            )


def render_contact_page(base_dir: str):
    """联系我们页面"""
    # 加载背景图片
    bg_path = os.path.join(base_dir, "data", "emailbackground.jpg")
    bg_base64 = ""
    if os.path.exists(bg_path):
        try:
            with open(bg_path, "rb") as f:
                bg_base64 = base64.b64encode(f.read()).decode()
        except Exception:
            pass

    # 注入背景样式（覆盖内页默认背景）
    bg_css = ""
    if bg_base64:
        bg_css = f"background-image: url(data:image/jpeg;base64,{bg_base64}); background-size: cover; background-repeat: no-repeat; background-position: center; background-attachment: fixed;"
    else:
        bg_css = "background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);"

    st.markdown(
        f"""
        <style>
        .stApp {{
            {bg_css}
        }}
        .contact-page-title {{
            color: #ffffff !important;
            text-align: center !important;
            text-shadow: 0 1px 4px rgba(0, 0, 0, 0.35);
        }}
        /* 确保内页内容区文字白色 */
        main .block-container,
        main .block-container * {{
            color: white !important;
        }}
        /* 按钮等交互元素颜色恢复深色 */
        main .block-container button,
        main .block-container .stButton button,
        main .block-container input,
        main .block-container textarea {{
            color: #2c3e50 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<h1 class="contact-page-title">📬 联系我们</h1>', unsafe_allow_html=True)

    # 返回首页按钮（居中）
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🏠 返回首页", use_container_width=True):
            st.session_state.current_page = "home"
            st.rerun()

    # 流星雨特效
    st.markdown(
        """
        <style>
        .meteor-container {
            position: fixed;
            inset: 0;
            pointer-events: none;
            overflow: hidden;
            z-index: 0;
        }
        .meteor {
            position: absolute;
            width: 2px;
            height: 90px;
            background: linear-gradient(to bottom, rgba(255,255,255,0.95), rgba(255,255,255,0));
            transform: rotate(35deg);
            filter: drop-shadow(0 0 6px rgba(255,255,255,0.8));
            animation: meteor-fall linear infinite;
        }
        @keyframes meteor-fall {
            0% {
                transform: translate(0, 0) rotate(35deg);
                opacity: 0;
            }
            10% { opacity: 1; }
            100% {
                transform: translate(-420px, 620px) rotate(35deg);
                opacity: 0;
            }
        }
        </style>
        <div class="meteor-container">
            <span class="meteor" style="left: 92%; top: 4%; animation-duration: 3.6s; animation-delay: 0s;"></span>
            <span class="meteor" style="left: 80%; top: 12%; animation-duration: 4.2s; animation-delay: 0.8s;"></span>
            <span class="meteor" style="left: 70%; top: 2%; animation-duration: 3.2s; animation-delay: 1.6s;"></span>
            <span class="meteor" style="left: 60%; top: 16%; animation-duration: 4.8s; animation-delay: 2.1s;"></span>
            <span class="meteor" style="left: 50%; top: 6%; animation-duration: 3.9s; animation-delay: 2.7s;"></span>
            <span class="meteor" style="left: 40%; top: 10%; animation-duration: 4.5s; animation-delay: 3.4s;"></span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 白色文字，居中显示（无容器）
    st.markdown(
        """
        <div style="text-align: center; font-size: 1.2rem; margin-top: 40px; color: white;">
            <p>感谢您对临床启明星的关注与支持！</p>
            <p>如果您希望学习某个疾病，或对我们的平台有任何建议，</p>
            <p>欢迎通过以下邮箱联系我们：</p>
            <p style="font-size: 1.5rem; font-weight: bold; margin: 1.5rem 0; color: white;">📧 1360557120@qq.com</p>
            <p>我们会认真阅读每一封来信，不断优化平台！</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_case_list(df_case: pd.DataFrame):
    st.markdown(inner_page_style(), unsafe_allow_html=True)
    st.markdown('<h1 style="color: white !important;">✨ 医小星深度精讲</h1>', unsafe_allow_html=True)

    if st.button("🔙 返回首页", key="back_home_case_list"):
        st.session_state.current_page = "home"
        st.rerun()

    st.markdown("### 请选择一个疾病，医小星将带你层层深入")

    systems = df_case["系统"].unique()
    for sys in systems:
        st.markdown(f"#### {sys}")
        diseases = df_case[df_case["系统"] == sys]["疾病"].unique()
        if len(diseases) == 0:
            continue
        cols = st.columns(min(len(diseases), 3))
        for idx, disease in enumerate(diseases):
            with cols[idx % 3]:
                if st.button(f"{disease}", key=f"case_{sys}_{disease}", use_container_width=True):
                    st.session_state.case_disease = disease
                    st.session_state.case_system = sys
                    st.session_state.case_step_idx = 0
                    st.session_state.case_messages = []
                    st.session_state.case_completed_steps = set()
                    st.session_state.current_page = "case_tutorial"
                    st.rerun()
        st.markdown("---")


def render_case_tutorial(df_case: pd.DataFrame, base_dir: str):
    st.markdown(inner_page_style(), unsafe_allow_html=True)
    disease = str(st.session_state.get("case_disease", "") or "").strip()
    system = str(st.session_state.get("case_system", "") or "").strip()

    # 加载医小星图片
    img_path = os.path.join(base_dir, "data", "images", MEDSTAR_ICON_FILENAME)
    img_base64 = _load_image_base64(img_path)
    if img_base64:
        title_icon = (
            f'<img src="{img_base64}" style="width: {MEDSTAR_ICON_SIZE}px; height: {MEDSTAR_ICON_SIZE}px; '
            'border-radius: 50%; vertical-align: middle; margin-right: 8px;">'
        )
        chat_avatar = (
            f'<img src="{img_base64}" style="width: 30px; height: 30px; border-radius: 50%; '
            'vertical-align: middle; margin-right: 4px;">'
        )
    else:
        title_icon = "🧑‍🏫"
        chat_avatar = "🧑‍🏫"

    st.markdown(
        f'<h1 style="color: white !important;">{title_icon} {disease}深度精讲</h1>',
        unsafe_allow_html=True,
    )

    if st.button("🔙 返回精讲列表", key="back_to_case_list"):
        st.session_state.current_page = "case_list"
        st.rerun()

    steps_df = df_case[(df_case["疾病"] == disease) & (df_case["系统"] == system)].sort_values(
        "步骤序号"
    )
    if steps_df.empty:
        st.error("未找到该疾病的精讲配置")
        return

    if "case_step_idx" not in st.session_state:
        st.session_state.case_step_idx = 0
    if "case_messages" not in st.session_state:
        st.session_state.case_messages = []
    if "case_completed_steps" not in st.session_state:
        st.session_state.case_completed_steps = set()

    total_steps = len(steps_df)

    if st.session_state.case_step_idx == 0:
        st.markdown("### 👋 欢迎来到医小星深度精讲")
        st.markdown(
            f"我们将一步步深入讲解 **{disease}** 的病因、病理、诊疗和康复。\n\n"
            "点击下方按钮开始学习，过程中你可以随时提问。"
        )
        if st.button("开始学习", key="start_case"):
            st.session_state.case_step_idx = 1
            st.rerun()
        return

    current_step = steps_df.iloc[st.session_state.case_step_idx - 1]
    step_num = current_step["步骤序号"]
    step_name = current_step["步骤名称"]
    prompt_template = current_step["讲解提示词模板"]

    step_icon = chat_avatar if img_base64 else "🧑‍🏫"
    st.markdown(f"### {step_icon} 第{step_num}步：{step_name}", unsafe_allow_html=True)

    for msg in st.session_state.case_messages:
        if msg["role"] == "assistant":
            with st.chat_message("assistant"):
                st.markdown(f'{chat_avatar} {msg["content"]}', unsafe_allow_html=True)
        else:
            with st.chat_message("user"):
                st.markdown(msg["content"])

    if step_num not in st.session_state.case_completed_steps:
        prompt = str(prompt_template).replace("{疾病}", disease)
        reply = call_general_ai(prompt)
        st.session_state.case_messages.append({"role": "assistant", "content": reply, "step": step_num})
        st.session_state.case_completed_steps.add(step_num)
        st.rerun()

    user_input = st.chat_input("你可以提问或回答上面的问题...")
    if user_input:
        st.session_state.case_messages.append({"role": "user", "content": user_input})
        feedback_prompt = (
            f"基于当前教学内容和学生的问题/回答：{user_input}，请你作为医小星给予反馈，并继续引导学习。"
            "如果学生回答正确，表扬并建议进入下一步；如果错误，解释后再问一次。"
        )
        hist = [{"role": m["role"], "content": m["content"]} for m in st.session_state.case_messages[-10:]]
        reply = call_general_ai(feedback_prompt, history=hist)
        st.session_state.case_messages.append({"role": "assistant", "content": reply})
        st.rerun()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("➡️ 下一步", key="next_step"):
            if st.session_state.case_step_idx < total_steps:
                st.session_state.case_step_idx += 1
                st.rerun()
    with col2:
        if st.button("🔁 重新开始", key="restart_case"):
            st.session_state.case_step_idx = 0
            st.session_state.case_messages = []
            st.session_state.case_completed_steps = set()
            st.rerun()

    if st.session_state.case_step_idx == total_steps:
        st.info("🎉 恭喜你完成了所有步骤！")
        if st.button("📄 生成学习小结", key="gen_summary"):
            summary_prompt = str(steps_df.iloc[-1]["讲解提示词模板"]).replace("{疾病}", disease)
            summary = call_general_ai(summary_prompt)
            st.session_state.case_messages.append(
                {"role": "assistant", "content": f"**学习小结**\n\n{summary}"}
            )
            st.rerun()


# 医小星情景交代消息（仅背景与已有检查结果，不包含症状描述；症状由患者在对话中说出）
NARRATOR_MESSAGES: dict[int, str] = {
    0: "张先生，49岁，保险公司经理，有长期吸烟饮酒史，高血压病史5年未规律服药。今天早上被家属送来医院。你是接诊医生，请开始问诊。",
    1: "CT结果已出，显示左基底节区高密度影，考虑脑出血。血脂检查也显示甘油三酯偏高。现在患者很担心，请你向他解释检查结果。",
    2: "患者目前病情稳定，家属很关心病情严重程度和预后。请你用患者能听懂的话解释。",
    3: "患者即将出院，需要交代后续治疗、康复和预防措施。请你给出综合建议。",
}


def _get_narrator_message(situation_index: int) -> str:
    """根据情境索引（0-based）返回医小星情景交代文案。"""
    return NARRATOR_MESSAGES.get(situation_index, NARRATOR_MESSAGES[0])


def _reset_simulation_state(situations: list, selected_sit) -> None:
    """重置模拟问诊与引导相关状态（情景交代仅在页面顶部显示，不写入 sim_messages）。"""
    st.session_state.sim_messages = []
    st.session_state.guide_messages = []
    st.session_state.guide_mode = False
    st.session_state.session_started = False
    st.session_state.guide_context = {}
    st.session_state.guide_completed = False
    st.session_state.show_guide_reminder = False
    st.session_state.bg_guide_done = False
    # st.session_state.symptom_guide_done = False  # 暂时注释，测试返回后按钮是否仍存在
    st.session_state.last_guided_message = ""
    st.session_state.guide_in_progress = False


def render_guide_mode(base_dir: str) -> None:
    """
    渲染医小星引导模式：独立聊天界面，返回问诊按钮，[DONE] 后显示返回。
    [修改] 增加 base_dir 参数，用于加载 MEDSTAR_ICON_FILENAME 作为 3D 医小星图标。
    """
    st.markdown(inner_page_style(), unsafe_allow_html=True)

    # ========== 加载医小星图片（用于标题和聊天头像） ==========
    img_path = os.path.join(base_dir, "data", "images", MEDSTAR_ICON_FILENAME)
    img_base64 = _load_image_base64(img_path)
    if st.button("🔙 返回问诊", key="back_to_patient_from_guide"):
        st.session_state.guide_mode = False
        st.rerun()

    if img_base64:
        title_icon = (
            f'<img src="{img_base64}" '
            f'style="width: {MEDSTAR_ICON_SIZE}px; height: {MEDSTAR_ICON_SIZE}px; border-radius: 50%; '
            f'vertical-align: middle; margin-right: 8px;">'
        )
        chat_avatar = (
            f'<img src="{img_base64}" '
            f'style="width: 30px; height: 30px; border-radius: 50%; vertical-align: middle; margin-right: 4px;">'
        )
    else:
        title_icon = "🧑‍🏫"
        chat_avatar = "🧑‍🏫"
        st.warning("医小星图片加载失败，将使用默认 emoji")

    st.markdown(
        f'<h3 style="color: white !important;">{title_icon} 医小星引导助手</h3>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="color: white !important; font-size: 0.9rem;">根据刚才患者的回答进行思考与讨论，完成后可点击上方返回问诊。</div>',
        unsafe_allow_html=True,
    )

    # 重新开始引导按钮：清空历史并刷新页面
    if st.button("🔄 重新开始引导", key="restart_guide"):
        st.session_state.guide_messages = []
        st.session_state.guide_completed = False
        st.rerun()

    if "guide_messages" not in st.session_state or not isinstance(
        st.session_state.guide_messages, list
    ):
        st.session_state.guide_messages = []

    # 如果引导历史为空，显示“开始引导”按钮，而不是自动生成消息
    if not st.session_state.guide_messages:
        ctx = st.session_state.get("guide_context") or {}
        if ctx:
            # 显示开始引导按钮
            if st.button("✨ 开始引导", key="start_guide"):
                current_grade = st.session_state.get("grade", "初级")
                first_reply = call_medstar(
                    [],
                    ctx,
                    student_level=current_grade,
                )
                st.session_state.guide_messages = [
                    {"role": "assistant", "content": first_reply},
                ]
                if "[DONE]" in first_reply:
                    st.session_state.guide_completed = True
                    if ctx.get("patient_recent"):
                        st.session_state.symptom_guide_done = True
                st.rerun()
        else:
            st.info("暂无引导上下文，请从问诊页面进入引导。")
        return  # 没有历史时，不显示对话历史

    # 有引导历史时，显示对话
    for msg in st.session_state.guide_messages:
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        display_content = (content.replace("[DONE]", "").strip() if content else "")
        if role == "user":
            with st.chat_message("user"):
                st.markdown(f'<p style="color: white !important;">👨‍🎓 <strong>学生</strong>：{display_content}</p>', unsafe_allow_html=True)
        else:
            with st.chat_message("assistant"):
                st.markdown(f'<p style="color: white !important;">{chat_avatar} <strong>医小星</strong>：{display_content}</p>', unsafe_allow_html=True)

    # 检查最后一条消息是否包含 [DONE] 标记
    if st.session_state.guide_messages:
        last = st.session_state.guide_messages[-1]
        if last.get("role") == "assistant" and "[DONE]" in last.get("content", ""):
            st.session_state.guide_completed = True
            if st.session_state.get("guide_context", {}).get("patient_recent"):
                st.session_state.symptom_guide_done = True
    if st.session_state.get("guide_completed", False):
        if st.button("🔙 返回问诊", key="auto_back_to_patient"):
            st.session_state.guide_mode = False
            st.rerun()

    guide_input = st.chat_input("请输入你的回答...")
    if guide_input:
        st.session_state.guide_messages.append(
            {"role": "user", "content": guide_input}
        )
        ctx = st.session_state.get("guide_context") or {}
        current_grade = st.session_state.get("grade", "初级")
        reply = call_medstar(
            st.session_state.guide_messages,
            ctx,
            student_level=current_grade,
        )
        st.session_state.guide_messages.append(
            {"role": "assistant", "content": reply}
        )
        if "[DONE]" in reply:
            st.session_state.guide_completed = True
            if st.session_state.get("guide_context", {}).get("patient_recent"):
                st.session_state.symptom_guide_done = True
        st.rerun()


def render_general_chat(base_dir: str) -> None:
    """渲染通用智能问答界面，用户可问任意医学问题，AI 回答。"""
    st.markdown(inner_page_style(), unsafe_allow_html=True)

    # 加载医小星图片（用于标题与头像）
    img_path = os.path.join(base_dir, "data", "images", MEDSTAR_ICON_FILENAME)
    img_base64 = _load_image_base64(img_path)
    if img_base64:
        title_icon = (
            f'<img src="{img_base64}" '
            f'style="width: {MEDSTAR_ICON_SIZE}px; height: {MEDSTAR_ICON_SIZE}px; border-radius: 50%; '
            f'vertical-align: middle; margin-right: 8px;">'
        )
        chat_avatar = (
            f'<img src="{img_base64}" '
            f'style="width: 30px; height: 30px; border-radius: 50%; vertical-align: middle; margin-right: 4px;">'
        )
    else:
        title_icon = "🧑‍🏫"
        chat_avatar = "🧑‍🏫"

    # 返回按钮：回到进入前页面（默认回首页）
    if st.button("🔙 返回", key="back_from_general"):
        prev = st.session_state.get("general_chat_prev", {}) or {}
        st.session_state.general_chat_mode = False
        st.session_state.current_page = prev.get("page", "home")
        st.session_state.current_system = prev.get("system", st.session_state.get("current_system"))
        st.session_state.current_disease = prev.get("disease", st.session_state.get("current_disease"))
        st.rerun()

    st.markdown(
        f'<h3 style="color: white !important;">{title_icon} 医小星智能问答</h3>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="color: white !important; font-size: 0.9rem;">你可以问我任何医学问题，我会尽力解答。</div>',
        unsafe_allow_html=True,
    )

    if "general_messages" not in st.session_state or not isinstance(
        st.session_state.general_messages, list
    ):
        st.session_state.general_messages = []

    # 显示聊天历史
    for msg in st.session_state.general_messages:
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        if role == "user":
            with st.chat_message("user"):
                st.markdown(
                    f'<p style="color: white !important;">👤 <strong>你</strong>：{content}</p>',
                    unsafe_allow_html=True,
                )
        else:
            with st.chat_message("assistant"):
                st.markdown(
                    f'<p style="color: white !important;">{chat_avatar} <strong>医小星</strong>：{content}</p>',
                    unsafe_allow_html=True,
                )

    user_input = st.chat_input("请输入你的问题...")
    if user_input:
        st.session_state.general_messages.append({"role": "user", "content": user_input})
        # 将历史（不含 system）传给通用模型，保持上下文连贯
        history = [
            {"role": m.get("role", "user"), "content": m.get("content", "")}
            for m in st.session_state.general_messages[-12:]
            if m.get("role") in ("user", "assistant")
        ]
        ai_reply = call_general_ai(user_input, history=history)
        st.session_state.general_messages.append({"role": "assistant", "content": ai_reply})
        st.rerun()


def render_patient_mode(
    situations: list,
    selected_sit,
    profile_row,
    sit_index: int,
    situation_num: int,
    base_dir: str,
) -> None:
    """
    患者对话模式：顶部情景交代与「让医小星来帮帮你吧」（背景教学）；患者消息下首次症状描述显示「再让医小星来帮帮你吧」一次，
    症状分析引导完成后每条患者消息下显示「需不需要医小星再帮帮忙？」；进入下一情境与重置同情境切换。
    [修改] 增加 base_dir 参数，用于在情景交代右侧按钮旁显示 3D 医小星图片。
    """
    if "sim_messages" not in st.session_state or not isinstance(
        st.session_state.sim_messages, list
    ):
        st.session_state.sim_messages = []

    # ========== 加载医小星图片（用于顶部按钮旁图标） ==========
    img_path = os.path.join(base_dir, "data", "images", MEDSTAR_ICON_FILENAME)
    img_base64 = _load_image_base64(img_path)

    # 优先从情景交代文件读取
    scene_df = st.session_state.get("scene_narrator", pd.DataFrame())
    if not scene_df.empty and "疾病" in scene_df.columns and "情境" in scene_df.columns and "情景交代" in scene_df.columns:
        current_disease = st.session_state.get("current_disease")
        if current_disease is not None:
            try:
                mask = (
                    (scene_df["疾病"].astype(str).str.strip() == str(current_disease).strip())
                    & (scene_df["情境"].astype(int) == sit_index + 1)
                )
                matches = scene_df[mask]
                if not matches.empty:
                    val = matches.iloc[0]["情景交代"]
                    if pd.notna(val) and str(val).strip():
                        narrator_text = str(val).strip()
                    else:
                        narrator_text = _get_narrator_message(sit_index)
                else:
                    narrator_text = _get_narrator_message(sit_index)
            except Exception:
                narrator_text = _get_narrator_message(sit_index)
        else:
            narrator_text = _get_narrator_message(sit_index)
    else:
        narrator_text = _get_narrator_message(sit_index)

    # 顶部：情景交代 + 引导按钮（背景教学仅显示一次，点击后隐藏）
    col_narrator, col_btn = st.columns([4, 1])
    with col_narrator:
        st.info(f"📋 **情景交代**：{narrator_text}")
    with col_btn:
        # [修改] 左侧医小星图标 + 右侧「问问医小星」按钮
        icon_html = ""
        if img_base64:
            icon_html = (
                f'<img src="{img_base64}" '
                f'style="width: {MEDSTAR_ICON_SIZE}px; height: {MEDSTAR_ICON_SIZE}px; border-radius: 50%; '
                f'vertical-align: middle; margin-right: 6px;">'
            )
        else:
            # 只警告一次，避免重复刷屏
            if not st.session_state.get("yixiaoxing_icon_warned", False):
                st.warning(f"未找到医小星图标：{img_path}，将使用默认 emoji。")
                st.session_state.yixiaoxing_icon_warned = True

        left, right = st.columns([1, 3])
        with left:
            if icon_html:
                st.markdown(icon_html, unsafe_allow_html=True)
            else:
                st.markdown("🧑‍🏫")
        with right:
            if not st.session_state.get("bg_guide_done", False):
                if st.button("问问医小星", key="medstar_top", use_container_width=True):
                    st.session_state.guide_context = {
                        "profile": profile_row,
                        "situation": situation_num,
                    }
                    st.session_state.guide_messages = []
                    st.session_state.guide_mode = True
                    st.session_state.guide_completed = False
                    st.session_state.bg_guide_done = True
                    st.rerun()

    # 未开始时显示「开始对话」
    if not st.session_state.get("session_started", False):
        if st.button("开始对话", key="start_simulation_chat"):
            st.session_state.session_started = True
            st.rerun()

    # 对话列表：仅 user / assistant（不显示 narrator）
    for idx, msg in enumerate(st.session_state.sim_messages):
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        if role == "narrator":
            continue
        if role == "user":
            with st.chat_message("user"):
                st.markdown(f'<p style="color: white !important;">👨‍⚕️ <strong>医生</strong>：{content}</p>', unsafe_allow_html=True)
        elif role == "assistant":
            with st.chat_message("assistant"):
                st.markdown(f'<p style="color: white !important;">👤 <strong>患者</strong>：{content}</p>', unsafe_allow_html=True)

    if st.session_state.get("session_started", False):
        user_input = st.chat_input("请输入你要向患者询问的问题...")
        if user_input:
            st.session_state.sim_messages.append(
                {"role": "user", "content": user_input}
            )
            ai_reply = call_ai_patient(st.session_state.sim_messages, profile_row)
            st.session_state.sim_messages.append(
                {"role": "assistant", "content": ai_reply}
            )
            st.rerun()

    # 始终显示帮助按钮（只要对话已经开始）
    if st.session_state.get("session_started", False):
        last_patient_msg = ""
        for msg in reversed(st.session_state.sim_messages):
            if msg.get("role") == "assistant":
                last_patient_msg = msg.get("content", "")
                break
        if not last_patient_msg:
            last_patient_msg = "患者还没有说话，请先开始问诊..."

        st.markdown("---")

        # [修改] 底部帮助区域：使用 flex 布局，让 40px 医小星图标与按钮紧密排列
        icon_html = (
            f'<img src="{img_base64}" alt="医小星">'
            if img_base64
            else '<span style="font-size: 32px;">🧑‍🏫</span>'
        )
        html_content = f"""
        <style>
        .bottom-help {{
            display: flex;
            justify-content: flex-end;
            align-items: center;
            gap: 4px;
            margin-top: 10px;
        }}
        .bottom-help img {{
            width: 40px;
            height: 40px;
            border-radius: 50%;
            object-fit: cover;
        }}
        .bottom-help button,
        .bottom-help .stButton button {{
            background-color: #4f8bf9 !important;
            color: white !important;
            border-radius: 8px !important;
            padding: 0.5rem 1rem !important;
            font-weight: 600 !important;
            border: none !important;
            box-shadow: 0 2px 6px rgba(0,0,0,0.08) !important;
            transition: background-color 0.2s !important;
        }}
        .bottom-help button:hover,
        .bottom-help .stButton button:hover {{
            background-color: #004999 !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.18) !important;
        }}
        </style>
        <div class="bottom-help">
            {icon_html}
            <div>
        """
        st.markdown(html_content, unsafe_allow_html=True)

        # 使用列只承载按钮组件，实际布局由外层 flex 容器控制
        col_btn = st.columns([3, 1])[1]
        with col_btn:
            if st.button(
                "需要我帮忙分析吗？",
                key="permanent_help_button",
                use_container_width=True,
            ):
                if last_patient_msg and "还没有说话" not in last_patient_msg:
                    st.session_state.guide_context = {
                        "patient_recent": last_patient_msg,
                        "profile": profile_row,
                        "situation": situation_num,
                    }
                    # 不主动清空 guide_messages，保留历史；需重新开始时在引导页点「重新开始引导」
                    st.session_state.guide_mode = True
                    st.rerun()
                else:
                    st.info("请先和患者对话，获取一些信息后再来找我哦～")

        st.markdown("</div></div>", unsafe_allow_html=True)

    # 情境完成提示（放在页面最底部、重置按钮上方）
    situation_completed = False
    for msg in reversed(st.session_state.sim_messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", "") or ""
            # 只要患者说了“谢谢医生”就认为该情境对话已完成
            if "谢谢医生" in content:
                situation_completed = True
            break
    if situation_completed:
        st.markdown("---")
        if sit_index + 1 < len(situations):
            next_sit_num = sit_index + 2
            st.success(f"✅ 情境{next_sit_num - 1}已完成，请从顶部的下拉框选择情境{next_sit_num}继续学习。")
        else:
            st.balloons()
            st.success("🎉 恭喜你完成了所有情境的学习！")

    if st.button("重置对话", key="reset_simulation_chat"):
        _reset_simulation_state(situations, selected_sit)
        st.rerun()


def render_simulation(df_patients: pd.DataFrame) -> None:
    """
    模拟问诊页面：患者对话模式 + 医小星引导模式。
    情境切换后显示情景交代，点击「开始对话」后可问诊；每条患者消息下可「让医小星来帮帮你吧」进入引导；可进入下一情境。
    [修改] 在调用 render_guide_mode / render_patient_mode 时传入 base_dir，用于加载 3D 医小星图片。
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))

    st.markdown(inner_page_style(), unsafe_allow_html=True)
    st.markdown('<h1 style="color: white !important;">临床启明星 - 模拟问诊</h1>', unsafe_allow_html=True)

    # 返回导航
    col_back_learn, col_back_disease, col_back_home = st.columns(3)
    with col_back_learn:
        if st.button("返回学习页", key="back_to_learn_from_sim"):
            st.session_state.current_page = "learn"
            st.rerun()
    with col_back_disease:
        if st.button("返回疾病列表", key="back_to_disease_from_sim"):
            go_diseases()
    with col_back_home:
        if st.button("返回首页", key="back_to_home_from_sim"):
            go_home()

    current_disease = st.session_state.get("current_disease")
    if not current_disease:
        st.info("当前未选择疾病，无法进入模拟问诊。请在疾病列表页选择疾病。")
        return

    st.markdown(f'<h2 style="color: white !important;">当前疾病：{current_disease}</h2>', unsafe_allow_html=True)

    if df_patients is None or df_patients.empty:
        st.info("病人设定文件为空或未正确加载，请联系管理员检查数据文件。")
        return

    df_dis = df_patients[df_patients["疾病"] == current_disease]
    if df_dis.empty:
        st.info("病人设定文件中未找到当前疾病的相关情境数据。")
        return

    if "情境编号" not in df_dis.columns:
        st.error("病人设定文件中缺少“情境编号”列，请检查 Excel 列名。")
        return

    situations = sorted(df_dis["情境编号"].dropna().unique().tolist())
    if not situations:
        st.info("病人设定文件中未找到任何情境编号，请检查“情境编号”列数据是否完整。")
        return

    default_sit = st.session_state.get("sim_situation", situations[0])
    if default_sit not in situations:
        default_sit = situations[0]

    # [修改] 情境切换方式改为按钮式单选（类似学习页的年级切换），提升可见性与可点性
    selected_sit = st.radio(
        "请选择情境编号",
        options=situations,
        index=situations.index(default_sit),
        horizontal=True,
        key="simulation_situation_selector",
    )

    # 下拉切换情境：重置状态（情景交代仅在页面顶部显示，不写入 sim_messages）
    if selected_sit != st.session_state.get("sim_situation"):
        st.session_state.sim_situation = selected_sit
        _reset_simulation_state(situations, selected_sit)

    df_sit = df_dis[df_dis["情境编号"] == selected_sit]
    if df_sit.empty:
        st.info("当前情境下未找到病人设定数据。")
        return

    profile_row = df_sit.iloc[0]
    sit_index = situations.index(selected_sit)
    situation_num = sit_index + 1  # 1/2/3/4

    if st.session_state.get("guide_mode", False):
        render_guide_mode(base_dir)
    else:
        render_patient_mode(
            situations, selected_sit, profile_row, sit_index, situation_num, base_dir
        )


def main():
    # 页面基础配置
    st.set_page_config(
        page_title="临床启明星",
        layout="wide",
        menu_items={
            "Get Help": None,
            "Report a bug": None,
            "About": None,
        },
    )

    # [修改] 注入 PWA 相关的 HTML 头部信息与 Service Worker 注册脚本，支持手机端安装到主屏幕
    st.markdown(
        """
        <link rel="manifest" href="/manifest.json">
        <meta name="apple-mobile-web-app-capable" content="yes">
        <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
        <meta name="apple-mobile-web-app-title" content="临床启明星">
        <meta name="theme-color" content="#0A2F6C">
        <script>
        if ('serviceWorker' in navigator) {
            window.addEventListener('load', function() {
                navigator.serviceWorker.register('/service-worker.js').then(function(registration) {
                    console.log('ServiceWorker注册成功：', registration.scope);
                }, function(err) {
                    console.log('ServiceWorker注册失败：', err);
                });
            });
        }
        </script>
        """,
        unsafe_allow_html=True,
    )

    # 计算数据文件路径（使用 os.path 保证跨平台）
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")

    # 先初始化 session，以便按 current_page / current_system 注入对应背景
    init_session_state()

    mcq_path = os.path.join(data_dir, "PBL题库.xlsx")
    short_q_path = os.path.join(data_dir, "简答题题库.xlsx")
    scoring_path = os.path.join(data_dir, "简答题评分.xlsx")
    patients_path = os.path.join(data_dir, "病人设定.xlsx")
    teaching_outlines_path = os.path.join(data_dir, "医小星教学大纲.xlsx")

    # 悬浮医小星图标触发：?general_chat=true → 进入通用问答模式（优先级最高）
    if st.query_params.get("general_chat") == "true":
        st.session_state.general_chat_prev = {
            "page": st.session_state.get("current_page", "home"),
            "system": st.session_state.get("current_system"),
            "disease": st.session_state.get("current_disease"),
        }
        # 兼容性：按指令额外保存 previous_* 字段
        st.session_state.previous_page = st.session_state.get("current_page", "home")
        st.session_state.previous_system = st.session_state.get("current_system")
        st.session_state.previous_disease = st.session_state.get("current_disease")
        st.session_state.general_chat_mode = True
        st.session_state.current_page = "general_chat"
        try:
            st.query_params.clear()
        except Exception:
            pass
        st.rerun()

    # 悬浮医小星图标触发：?guide=true → 进入引导模式（优先级高于 ?system）
    if st.query_params.get("guide") == "true":
        st.session_state.guide_mode = True
        patient_msg = st.query_params.get("msg", "你好，医小星，我想了解一下医学知识。")
        dummy_profile = pd.Series(
            {
                "疾病": "通用咨询",
                "年龄": "通用",
                "性别": "通用",
                "阶段描述": "通用咨询",
                "症状": "无",
                "体征": "无",
                "辅助检查": "无",
                "既往史": "无",
                "性格": "温和",
                "对话风格": "鼓励、亲切",
                "当前认知": "想了解医学知识",
                "关心问题": "医学知识",
            }
        )
        st.session_state.guide_context = {
            "patient_recent": patient_msg,
            "profile": dummy_profile,
            "situation": 1,
        }
        try:
            st.query_params.clear()
        except Exception:
            pass
        st.rerun()

    # 首页环形卡片点击通过 ?system=xxx 跳转：读到参数后写入 session、清除 URL 并 rerun
    if st.query_params.get("system"):
        st.session_state.current_system = st.query_params.get("system")
        st.session_state.current_page = "diseases"
        st.session_state.current_disease = None
        st.session_state.situation_state = {}
        try:
            st.query_params.clear()
        except Exception:
            pass
        st.rerun()

    # 按当前页面与系统注入背景（问问医小星开启时 current_page 可能仍为 home，需单独映射）
    page_for_bg = st.session_state.current_page
    if st.session_state.get("general_chat_mode", False) or page_for_bg == "general_chat":
        page_for_bg = "general_chat"
    _inject_page_background(base_dir, page_for_bg, st.session_state.get("current_system"))

    # [修改] 移除右下角悬浮按钮（改为首页标题按钮入口）
    # inject_floating_icon(base_dir)

    # 加载数据（若文件缺失将在对应函数中停止运行）
    df_mcq = load_mcq_bank(mcq_path)
    df_short_q = load_short_question_bank(short_q_path)
    df_scoring = load_short_question_scoring(scoring_path)
    df_patients = load_patient_profiles(patients_path)
    df_teaching_outlines = load_teaching_outlines(teaching_outlines_path)
    case_path = os.path.join(data_dir, "疾病精讲.xlsx")
    df_case = load_case_outlines(case_path)

    # 将教学大纲表放入 session_state，供 call_medstar 随时读取
    st.session_state.teaching_outlines = df_teaching_outlines

    # 加载情景交代文件
    scene_narrator_path = os.path.join(data_dir, "情景交代.xlsx")
    df_scene = load_scene_narrator(scene_narrator_path)
    st.session_state.scene_narrator = df_scene

    # 加载简答题参考答案
    ref_path = os.path.join(data_dir, "简答题参考答案.xlsx")
    if os.path.exists(ref_path):
        df_ref = pd.read_excel(ref_path)
        st.session_state.short_answer_refs = df_ref
    else:
        st.warning("未找到简答题参考答案文件，将不会显示参考答案。")
        st.session_state.short_answer_refs = pd.DataFrame()

    # 页面导航
    page = st.session_state.current_page
    if st.session_state.get("general_chat_mode", False) or page == "general_chat":
        render_general_chat(base_dir)
        return
    elif page == "home":
        render_home(df_mcq, base_dir)
    elif page == "diseases":
        render_diseases(df_mcq)
    elif page == "learn":
        render_learn(df_mcq, df_short_q, df_scoring)
    elif page == "report":
        render_report(df_mcq, df_short_q, df_scoring)
    elif page == "simulation":
        render_simulation(df_patients)
    elif page == "case_list":
        render_case_list(df_case)
    elif page == "case_tutorial":
        render_case_tutorial(df_case, base_dir)
    elif page == "contact":
        render_contact_page(base_dir)
    else:
        # 兜底：未知页面时回到首页
        st.session_state.current_page = "home"
        render_home(df_mcq, base_dir)


if __name__ == "__main__":
    main()


