import pandas as pd
import os

# 文件路径
data_dir = "data"
question_path = os.path.join(data_dir, "简答题题库.xlsx")
scoring_path = os.path.join(data_dir, "简答题评分.xlsx")
output_path = os.path.join(data_dir, "简答题参考答案.xlsx")

# 读取题库和评分表
df_q = pd.read_excel(question_path)
df_score = pd.read_excel(scoring_path)

# 清理列名（去除前后空格）
df_score.columns = df_score.columns.str.strip()
df_q.columns = df_q.columns.str.strip()

print("评分表的列名：", df_score.columns.tolist())
# 确认列名
if "关键词" not in df_score.columns:
    raise KeyError("评分表中没有'关键词'列，请检查列名是否正确。")
keyword_col = "关键词"
situation_col = "情境"
grade_col = "适用年级"

# 定义清洗函数，去除疾病名称中的引号和空格
def clean_name(name):
    if pd.isna(name):
        return ""
    s = str(name).strip().strip('"').strip("'")
    return s

# 对评分表进行清洗，方便匹配
df_score["疾病_clean"] = df_score["疾病"].apply(clean_name)

# 定义函数：根据疾病、情境、年级获取关键词列表
def get_keywords(disease, situation, grade):
    disease_clean = clean_name(disease)
    # 先按疾病筛选
    mask = df_score["疾病_clean"] == disease_clean
    if mask.sum() == 0:
        return []
    df_temp = df_score[mask].copy()

    # 按情境筛选（支持逗号分隔，如 "1,2"）
    sit_str = str(situation).strip()
    def match_situation(row_sit):
        if pd.isna(row_sit):
            return False
        parts = str(row_sit).split(",")
        parts = [p.strip() for p in parts]
        return sit_str in parts
    df_temp = df_temp[df_temp[situation_col].apply(match_situation)]

    # 按年级筛选（支持逗号分隔）
    grade_str = str(grade).strip()
    def match_grade(row_grade):
        if pd.isna(row_grade):
            return False
        parts = str(row_grade).split(",")
        parts = [p.strip() for p in parts]
        return grade_str in parts
    df_temp = df_temp[df_temp[grade_col].apply(match_grade)]

    # 获取关键词，去重，排序
    if df_temp.empty:
        return []
    keywords = df_temp[keyword_col].dropna().unique().tolist()
    return sorted(keywords)

# 为题库每一行生成参考答案
references = []
for idx, row in df_q.iterrows():
    disease = row["疾病"]
    situation = row["情境"]
    grade = row["年级等级"]
    keywords = get_keywords(disease, situation, grade)
    if keywords:
        ref = "、".join(keywords)
    else:
        ref = ""  # 无关键词则留空
    references.append(ref)

# 添加新列
df_q["参考答案"] = references

# 保存为新的Excel文件（只保留需要的列）
df_result = df_q[["疾病", "情境", "简答题题目", "年级等级", "参考答案"]]
df_result.to_excel(output_path, index=False)
print(f"✅ 生成完成！文件已保存至：{output_path}")