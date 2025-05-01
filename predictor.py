import re
import joblib
import numpy as np
import pandas as pd

# ------------------------
# 成分解析函数（和之前一样）
# ------------------------
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

# ------------------------
# 加载模型和工具
# ------------------------
def load_model():
    model = joblib.load('random_forest_model.pkl')
    scaler = joblib.load('minmax_scaler.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
    return model, scaler, feature_columns

# ------------------------
# 输入字符串 => 预测值
# ------------------------
def predict(model, scaler, feature_columns, composition_str):
    parsed = parse_composition(composition_str)
    input_df = pd.DataFrame([parsed], columns=feature_columns).fillna(0)

    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0.0
    input_df = input_df[feature_columns]

    input_scaled = scaler.transform(input_df)
    y_pred = model.predict(input_scaled)
    return float(y_pred[0])