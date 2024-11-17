import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用Agg后端，不显示图形而是保存图形
import matplotlib.pyplot as plt

# 读取csv数据，跳过第一行标题
data = pd.read_csv('processed_file_1014.csv', skiprows=1)

# 提取输入和输出
X = data.iloc[:, [7, 8]].values  # 第8列和第9列作为输入（幅值和相位）
y = data.iloc[:, 4].values  # 第5列作为输出（初始强度）

# 数据标准化
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)

# 根据比例手动划分训练集、验证集和测试集，而不打乱顺序
n_total = len(X)
n_train = int(0.7 * n_total)
n_val = int(0.15 * n_total)

X_train, y_train = X[:n_train], y[:n_train]
X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

# 转换为PyTorch的张量
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)


# 定义前馈神经网络
class FeedforwardNN(nn.Module):
    def __init__(self):
        super(FeedforwardNN, self).__init__()
        self.fc1 = nn.Linear(2, 128)  # 输入层：2个特征（幅值和相位）
        self.fc2 = nn.Linear(128, 64)  # 隐藏层1
        self.fc3 = nn.Linear(64, 32)  # 隐藏层2
        self.fc4 = nn.Linear(32, 1)  # 输出层：1个特征（初始强度）

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# 创建模型、损失函数和优化器
model = FeedforwardNN()
criterion = nn.MSELoss()  # 均方误差损失
optimizer = optim.AdamW(model.parameters(), lr=0.001)


# 训练模型并记录损失
def train_model(model, X_train, y_train, X_val, y_val, epochs=200):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')  # 初始化最佳验证损失
    early_stop_patience = 20  # 设置提前停止的耐心值
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # 前向传播
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        # 保存训练损失
        train_losses.append(loss.item())

        # 验证阶段
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            val_losses.append(val_loss.item())

        # 提前停止机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # 重置耐心计数器
            torch.save(model.state_dict(), '../deeplearning/best_model.pth')  # 保存模型
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        # 每10个epoch打印损失
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

    return train_losses, val_losses


# 调用训练函数
train_losses, val_losses = train_model(model, X_train, y_train, X_val, y_val, epochs=200)

# 加载最佳模型
model.load_state_dict(torch.load('../deeplearning/best_model.pth'))

# 测试模型
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs, y_test)
    print(f'Test Loss: {test_loss.item():.4f}')

plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.grid(True)
plt.savefig('loss_curve.png')  # 保存图形为 PNG 文件

