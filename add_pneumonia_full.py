# -*- coding: utf-8 -*-
"""
PBL助手 - 呼吸系统肺炎案例数据导入脚本
自动向 病人设定.xlsx、PBL题库.xlsx、简答题题库.xlsx、简答题评分.xlsx 添加肺炎案例数据。
若文件不存在则创建并写入列名与数据。
"""
from __future__ import print_function

import os
import sys

def ensure_deps():
    try:
        import pandas as pd  # noqa: F401
        import openpyxl  # noqa: F401
        return True
    except ImportError as e:
        print("缺少依赖库，请先运行： pip install pandas openpyxl")
        print("错误详情：", e)
        return False


def data_dir():
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "data")


# ---------- 病人设定.xlsx ----------
PATIENTS_COLUMNS = [
    "情境编号", "疾病", "阶段描述", "年龄", "性别", "症状", "体征", "辅助检查",
    "既往史", "性格", "对话风格", "当前认知", "关心问题", "标准诊断", "其他说明", "情景交代"
]
PATIENTS_DATA = """
1	肺炎	发病初期，门诊就诊	23	女	浑身酸软、头胀、发热（最高40℃）、咳嗽、咳少量白色黏痰、不易咳出、夜间咳嗽稍多、咳时上腹部疼痛、乏力、肌肉酸痛、头痛（发热时）、精神差、胃口差	暂无	尚未检查	无	焦虑	担心、急切	认为是普通感冒，但吃药无效	为什么感冒总不好？严不严重？	（未确诊）	旅游劳累史	李小姐，23岁，公司白领，旅游后出现发热、咳嗽，自行服药无效，来门诊就诊。
2	肺炎	门诊检查及初步治疗	23	女	咳嗽加剧，痰不多，余同前	体温38.3℃，呼吸16次/分，心率100次/分，血压90/60mmHg，两肺呼吸音粗糙，左下肺湿啰音	血常规：WBC 6.6, Neu% 73.5(H), Hb 119(L), PLT 112；胸片提示左下肺渗出	无	焦虑	担心治疗效果	知道做了检查，等待结果	检查结果如何？为什么用药没效果？	（未确诊）	头孢西丁治疗3天无效	门诊检查发现左下肺湿啰音，血常规显示中性粒细胞偏高，胸片提示肺炎可能，医生给予头孢西丁治疗，但3天后症状无改善。
3	肺炎	住院后病情加重	23	女	咳嗽加剧，痰量增多，左侧胸痛，气急	体温38.2℃，心率96次/分，呼吸18次/分，血压90/60mmHg，双上肢可见暗红色丘疹，有抓痕；左中肺呼吸音低，语颤增强，左肺底湿啰音	血常规正常；抗肺炎支原体抗体IgM 1:160阳性；CT左上肺大片感染；血气pH7.45, PO2 81, PCO2 39.2, HCO3 28.1；CRP 66.6，血沉58	蚊虫叮咬史	焦虑	紧张，担心病情	知道自己得了肺炎，但治疗效果不好	为什么越治越重？会不会有危险？	（未确诊）	头孢曲松+阿奇霉素治疗3天无效	患者住院后，检查发现肺炎支原体抗体阳性，CT显示左上肺感染，但抗生素治疗3天后病情反而加重，出现胸痛、气急。
4	肺炎	治疗后好转及出院指导	23	女	好转，轻微咳嗽	病情稳定	血气：pH7.47, PO2 68, PCO2 30.4, HCO3 20.5；左侧少量胸腔积液	同前	平静	感激，关心预防	知道病情好转，想出院	出院后注意事项，如何预防复发	肺炎（支原体肺炎）	母亲担心药物副作用	换用莫西沙星后，患者体温逐渐正常，症状好转。出院前，患者母亲担心药物副作用和预防措施，医生给予详细解答。
""".strip()


# ---------- PBL题库.xlsx ----------
MCQ_COLUMNS = [
    "系统", "疾病", "情境", "阶", "问题", "选项A", "选项B", "选项C", "选项D", "答案", "知识点", "年级等级"
]
MCQ_DATA = """
呼吸系统	肺炎	1	1	青年女性，急性起病，主要症状包括？	发热、咳嗽、咳痰	胸痛、气急	咯血、盗汗	恶心、呕吐	A	症状学	初级
呼吸系统	肺炎	1	1	患者体温最高达40℃，属于？	低热	中度发热	高热	超高热	C	体温分级	初级
呼吸系统	肺炎	1	1	患者自行服用退热药后体温可降至正常，说明？	病情好转	药物有效	病情自限	药物暂时抑制体温	D	药物作用	初级
呼吸系统	肺炎	1	1	患者咳嗽特点不包括？	单声咳	咳少量白色黏痰	夜间咳嗽较多	咳大量脓痰	D	症状学	初级
呼吸系统	肺炎	1	1	患者发病前的诱因最可能是？	受凉	劳累	淋雨	接触患者	B	病因学	初级
呼吸系统	肺炎	2	2	患者查体发现左下肺湿啰音，提示？	气道狭窄	气道内分泌物	胸腔积液	肺实变	B	体征	初级
呼吸系统	肺炎	2	2	患者血常规示Neu% 73.5%，提示？	病毒感染	细菌感染	支原体感染	真菌感染	B	实验室检查	初级
呼吸系统	肺炎	2	2	患者Hb 119g/L，属于？	正常	轻度贫血	中度贫血	重度贫血	B	实验室检查	初级
呼吸系统	肺炎	2	2	胸片提示左下肺渗出，最可能的诊断是？	支气管炎	肺炎	肺结核	肺癌	B	影像学	初级
呼吸系统	肺炎	2	2	患者血压90/60mmHg，心率100次/分，提示？	正常	休克早期	休克	高血压	B	生命体征	初级
呼吸系统	肺炎	3	3	患者双上肢出现暗红色丘疹，有抓痕，应考虑？	药物过敏	蚊虫叮咬	病毒感染	细菌感染	B	体征	中级
呼吸系统	肺炎	3	3	抗肺炎支原体抗体IgM 1:160阳性，提示？	近期感染	既往感染	疫苗接种	假阳性	A	实验室检查	中级
呼吸系统	肺炎	3	3	患者出现胸痛、气急，最可能的原因是？	肺炎加重	胸膜炎	气胸	心衰	B	并发症	中级
呼吸系统	肺炎	3	3	患者血气分析PO2 81mmHg，属于？	正常	轻度低氧	中度低氧	重度低氧	A	血气分析	中级
呼吸系统	肺炎	3	3	CRP 66.6mg/l，血沉58mm/h，提示？	细菌感染	病毒感染	真菌感染	非感染性炎症	A	实验室检查	中级
呼吸系统	肺炎	3	3	患者头孢曲松+阿奇霉素治疗3天无效，最可能的原因是？	药物剂量不足	病原体耐药	诊断错误	并发症出现	B	治疗	高级
呼吸系统	肺炎	4	4	换用莫西沙星后患者好转，提示病原体可能是？	肺炎链球菌	肺炎支原体	肺炎衣原体	军团菌	B	治疗	高级
呼吸系统	肺炎	4	4	患者出现左侧少量胸腔积液，属于？	漏出液	渗出液	脓胸	血胸	B	并发症	高级
呼吸系统	肺炎	4	4	患者出院时仍有轻微咳嗽，应建议？	继续口服抗生素	止咳药	观察	复查CT	C	出院指导	高级
呼吸系统	肺炎	4	4	患者母亲担心“下次无药好用”，正确的解释是？	会产生耐药	不必担心	需要轮换用药	需要联合用药	B	医患沟通	高级
呼吸系统	肺炎	4	4	预防肺炎的措施不包括？	接种疫苗	加强锻炼	避免劳累	长期服用抗生素	D	预防	高级
""".strip()


# ---------- 简答题题库.xlsx ----------
SHORT_Q_COLUMNS = ["疾病", "情境", "简答题题目", "年级等级"]
SHORT_Q_DATA = """
肺炎	1	根据情境1，请总结患者的临床表现，并分析可能的病因。	初级
肺炎	2	根据情境2的检查结果，请分析患者的诊断依据，并解释为什么头孢西丁治疗无效。	中级
肺炎	3	根据情境3，患者病情加重的原因可能是什么？结合检查结果分析。	高级
肺炎	4	请列出患者的完整诊断、治疗经过及出院指导。	高级
""".strip()


# ---------- 简答题评分.xlsx ----------
SCORING_COLUMNS = ["疾病", "关键词", "分值", "同义词1", "同义词2", "同义词3", "同义词4", "适用年级"]
SCORING_DATA = """
肺炎	发热	2	发烧	高热		初级
肺炎	咳嗽	2	咳痰		初级
肺炎	湿啰音	2	啰音		初级
肺炎	劳累	2	疲劳	旅游	诱因		初级
肺炎	支原体	3	肺炎支原体	MP		中级,高级
肺炎	抗体	3	IgM	血清学		中级,高级
肺炎	耐药	3	无效	不敏感		高级
肺炎	莫西沙星	3	莫西	抗生素		高级
肺炎	胸腔积液	3	胸水		高级
肺炎	预防	2	疫苗	接种		高级
""".strip()


def parse_tsv(text, columns):
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < len(columns):
            parts.extend([""] * (len(columns) - len(parts)))
        elif len(parts) > len(columns):
            parts = parts[:len(columns)]
        rows.append(parts)
    return rows


def ensure_data_dir():
    d = data_dir()
    if not os.path.exists(d):
        os.makedirs(d)
        print("已创建目录：", d)
    return d


def add_patients(path):
    ensure_data_dir()
    new_rows = parse_tsv(PATIENTS_DATA, PATIENTS_COLUMNS)
    new_df = __import__("pandas").DataFrame(new_rows, columns=PATIENTS_COLUMNS)
    if os.path.exists(path):
        df = __import__("pandas").read_excel(path)
        df = __import__("pandas").concat([df, new_df], ignore_index=True)
    else:
        df = new_df
    df.to_excel(path, index=False, engine="openpyxl")
    print("已写入/更新：", path, "（病人设定 肺炎 4 行）")


def add_mcq(path):
    ensure_data_dir()
    new_rows = parse_tsv(MCQ_DATA, MCQ_COLUMNS)
    new_df = __import__("pandas").DataFrame(new_rows, columns=MCQ_COLUMNS)
    if os.path.exists(path):
        df = __import__("pandas").read_excel(path)
        df = __import__("pandas").concat([df, new_df], ignore_index=True)
    else:
        df = new_df
    df.to_excel(path, index=False, engine="openpyxl")
    print("已写入/更新：", path, "（选择题 肺炎 22 行）")


def add_short_q(path):
    ensure_data_dir()
    new_rows = parse_tsv(SHORT_Q_DATA, SHORT_Q_COLUMNS)
    new_df = __import__("pandas").DataFrame(new_rows, columns=SHORT_Q_COLUMNS)
    if os.path.exists(path):
        df = __import__("pandas").read_excel(path)
        df = __import__("pandas").concat([df, new_df], ignore_index=True)
    else:
        df = new_df
    df.to_excel(path, index=False, engine="openpyxl")
    print("已写入/更新：", path, "（简答题 肺炎 4 行）")


def add_scoring(path):
    ensure_data_dir()
    new_rows = parse_tsv(SCORING_DATA, SCORING_COLUMNS)
    new_df = __import__("pandas").DataFrame(new_rows, columns=SCORING_COLUMNS)
    if os.path.exists(path):
        df = __import__("pandas").read_excel(path)
        df = __import__("pandas").concat([df, new_df], ignore_index=True)
    else:
        df = new_df
    df.to_excel(path, index=False, engine="openpyxl")
    print("已写入/更新：", path, "（简答题评分 肺炎 10 行）")


def main():
    if not ensure_deps():
        sys.exit(1)
    import pandas as pd  # noqa: F401
    d = data_dir()
    patients_path = os.path.join(d, "病人设定.xlsx")
    mcq_path = os.path.join(d, "PBL题库.xlsx")
    short_q_path = os.path.join(d, "简答题题库.xlsx")
    scoring_path = os.path.join(d, "简答题评分.xlsx")

    add_patients(patients_path)
    add_mcq(mcq_path)
    add_short_q(short_q_path)
    add_scoring(scoring_path)

    print("")
    print("呼吸系统肺炎案例数据已全部添加完成。")
    print("可在 PBL 助手中选择「肺炎」进行模拟问诊与学习。")


if __name__ == "__main__":
    main()
