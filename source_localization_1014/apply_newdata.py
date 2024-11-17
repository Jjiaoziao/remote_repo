import pandas as pd
import torch
import joblib  # 用于加载保存的scaler
import numpy as np

# 1. 加载已经保存的模型
class FeedforwardNN(torch.nn.Module):
    def __init__(self):
        super(FeedforwardNN, self).__init__()
        self.fc1 = torch.nn.Linear(2, 128)  # 输入层：2个特征（幅值和相位）
        self.fc2 = torch.nn.Linear(128, 64)  # 隐藏层1
        self.fc3 = torch.nn.Linear(64, 32)  # 隐藏层2
        self.fc4 = torch.nn.Linear(32, 1)  # 输出层：1个特征（初始强度）

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# 加载模型
model = FeedforwardNN()
model.load_state_dict(torch.load('../source_localization_1014/best_model.pth'))  # 加载模型
model.eval()  # 切换为推理模式

# 2. 读取新的数据文件 data.csv
new_data = pd.read_csv('finished_data_1012.csv', skiprows=1)

# 将科学计数法的列转换为浮点数，避免科学计数法格式
pd.options.display.float_format = '{:.8f}'.format  # 设置全局浮点数显示格式

# 提取第8列和第9列作为输入（假设新数据结构与训练数据一致）
X_new = new_data.iloc[:, [7, 8]].astype(float).values  # 将数据转换为浮点数格式

# 提取第5列作为真实值
y_true = new_data.iloc[:, 4].astype(float).values  # 将真实值提取为浮点数格式

# 3. 对新数据进行标准化（加载保存的Scaler）
scaler = joblib.load('scaler_1014.save')  # 加载训练时保存的scaler
X_new = scaler.transform(X_new)  # 使用相同的scaler进行标准化

# 转换为 PyTorch 张量
X_new = torch.tensor(X_new, dtype=torch.float32)

# 4. 使用模型对新数据进行推理
with torch.no_grad():
    new_predictions = model(X_new)

# 将预测结果转换为 numpy 格式
new_predictions = new_predictions.squeeze().numpy()

# 5. 保存预测结果和真实值到 CSV 文件
prediction_df = pd.DataFrame({
    'Predicted': new_predictions,
    'Actual': y_true
})

# 将结果保存为 CSV 文件
prediction_df.to_csv('new_data_predictions_with_actual.csv', index=False)

print("Predictions and actual values saved to 'new_data_predictions_with_actual.csv'")
