import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from torch.utils.data import DataLoader, TensorDataset

# 读取csv数据，跳过第一行标题
data = pd.read_csv('processed_file_1014.csv', skiprows=1)

# 提取输入和输出
X = data.iloc[:, [3, 7, 8]].values  # 第4列, 第8列和第9列作为输入
y = data.iloc[:, [1, 2]].values  # 第2列和第3列作为输出

# 数据标准化 - 对输入和输出进行标准化
scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)

# 保存scaler到文件，确保推理时使用相同的scaler
joblib.dump(scaler_X, 'scaler_X_1015.save')
joblib.dump(scaler_y, 'scaler_y_1015.save')

# 根据比例手动划分训练集、验证集和测试集，而不打乱顺序
n_total = len(X)
n_train = int(0.7 * n_total)
n_val = int(0.15 * n_total)

X_train, y_train = X[:n_train], y[:n_train]
X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

# 转换为PyTorch的张量
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # RNN 需要输入三维数据
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
y_val = torch.tensor(y_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test, dtype=torch.float32)

# 创建数据集和数据加载器
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

batch_size = 64  # 增加批量大小
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

# 定义一个简单的前馈神经网络 (MLP) 结构
class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

# 创建模型
input_size = 3  # 输入特征数
hidden_size = 128  # 减少隐藏层神经元数量
output_size = 2  # 输出特征数（2个坐标）
model = MLPModel(input_size, hidden_size, output_size)

# 损失函数使用MSE (均方误差)
criterion = nn.MSELoss()  # MSE 损失

# 优化器
optimizer = optim.Adam(model.parameters(), lr=0.00005)  # 调整学习率
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)  # 根据验证损失调整学习率

# 训练模型函数
def train_model(model, train_loader, val_loader, epochs=500):
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch.squeeze(1))  # 去掉多余维度以适应MLP
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for X_batch, y_batch in val_loader:
                val_outputs = model(X_batch.squeeze(1))  # 去掉多余维度以适应MLP
                val_loss += criterion(val_outputs, y_batch).item()

        scheduler.step(val_loss)  # 动态调整学习率

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')

# 调用训练函数
train_model(model, train_loader, val_loader, epochs=500)

# 加载最佳模型
model.load_state_dict(torch.load('best_model.pth_1015'))

# 测试模型
model.eval()
with torch.no_grad():
    test_outputs = model(X_test.squeeze(1))
    test_loss = criterion(test_outputs, y_test)
    print(f'Test Loss: {test_loss.item():.4f}')
