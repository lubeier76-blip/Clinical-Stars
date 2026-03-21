import pandas as pd
import os

# 文件路径
file_path = os.path.join("data", "PBL题库.xlsx")
output_path = os.path.join("data", "PBL题库_with_explanations.xlsx")

# 读取题库
df = pd.read_excel(file_path)

# 生成解析函数
def generate_explanation(row):
    answer = str(row["答案"]).strip()
    knowledge = str(row["知识点"]).strip() if pd.notna(row["知识点"]) else ""
    
    # 处理多选题（答案长度大于1，如"ABC"）
    if len(answer) > 1:
        # 将"ABC"转换为"A、B、C"
        ans_list = "、".join(list(answer))
        text = f"正确答案包括 {ans_list}。"
    else:
        text = f"正确答案是 {answer}。"
    
    if knowledge:
        text += f"（知识点：{knowledge}）"
    return text

# 应用函数生成解析列
df["解析"] = df.apply(generate_explanation, axis=1)

# 保存为新文件
df.to_excel(output_path, index=False)
print(f"✅ 已生成带解析的题库文件：{output_path}")