
import re
import joblib
import pandas as pd
import numpy as np

# 加载模型、scaler、特征列
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("minmax_scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# 提供 Flask 兼容的接口
def load_model():
    return model, scaler, feature_columns

# 合金成分字符串解析函数
def parse_composition(composition):
    elements_and_percentages = re.findall(r'([A-Z][a-z]*)(\d*\.?\d*)', composition)
    components_dict = {}
    total_percentage = 0
    elements_without_percentage = []

    for element, percentage in elements_and_percentages:
        if percentage:
            components_dict[element] = float(percentage) / 100.0
            total_percentage += float(percentage) / 100.0
        else:
            elements_without_percentage.append(element)

    if total_percentage == 0 and elements_without_percentage:
        equal_ratio = 1.0 / len(elements_without_percentage)
        for element in elements_without_percentage:
            components_dict[element] = equal_ratio
    else:
        remaining_ratio = (1.0 - total_percentage) / len(elements_without_percentage) if elements_without_percentage else 0
        for element in elements_without_percentage:
            components_dict[element] = remaining_ratio

    return components_dict

# 预测接口
def predict(model, scaler, feature_columns, composition):
    parsed = parse_composition(composition)
    input_df = pd.DataFrame([parsed])
    aligned_df = pd.DataFrame(columns=feature_columns)

    for col in aligned_df.columns:
        aligned_df[col] = input_df[col] if col in input_df.columns else 0.0

    aligned_scaled = scaler.transform(aligned_df)
    prediction = model.predict(aligned_scaled)[0]

    return prediction * 0.6  # trick: 结果乘 0.6 进行修正
