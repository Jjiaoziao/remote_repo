import pandas as pd

# 读取CSV文件，指定没有标题行，并手动设置列名
df = pd.read_csv('validation_comparison_ordered.csv', header=None, names=['Predicted', 'Actual'])

# 定义组大小
group_size = 10

# 创建三个空列表用于存储每组的序号、预测值最大值的序号、真实值最大值的序号
group_indices = []
predicted_max_indices = []
actual_max_indices = []
matches = []

# 遍历每10行作为一组，计算每组中最大值的序号
for i in range(0, len(df), group_size):
    group = df.iloc[i:i + group_size]

    # 获取当前组号
    group_index = i // group_size + 1
    group_indices.append(group_index)

    # 获取预测值的最大值序号
    predicted_max_idx = group['Predicted'].idxmax() - i + 1  # 加1以从1开始计数
    predicted_max_indices.append(predicted_max_idx)

    # 获取真实值的最大值序号
    actual_max_idx = group['Actual'].idxmax() - i + 1  # 加1以从1开始计数
    actual_max_indices.append(actual_max_idx)

    # 检查预测值和真实值的最大值序号是否匹配
    if predicted_max_idx == actual_max_idx:
        matches.append(1)  # 匹配则为1
    else:
        matches.append(0)  # 不匹配则为0

# 计算匹配的百分比
match_percentage = sum(matches) / len(matches) * 100

# 将结果保存为新的DataFrame
result_df = pd.DataFrame({
    'Group_Index': group_indices,
    'Predicted_Max_Index': predicted_max_indices,
    'Actual_Max_Index': actual_max_indices,
    'Match': matches
})

# 打印结果
print(result_df)
print(f"Percentage of matching groups: {match_percentage:.2f}%")

# 将结果保存为CSV文件
result_df.to_csv('max_index_comparison_with_groups.csv', index=False)

# 打印匹配百分比
print(f"Matching percentage: {match_percentage:.2f}%")
