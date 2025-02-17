import numpy as np
import matplotlib.pyplot as plt

# 定义 X 和 Y 的取值范围
x = np.array([1, 2, 3])
y = np.array([4, 5])

# 生成网格
X, Y = np.meshgrid(x, y)

print("X 坐标矩阵：")
print(X)

print("\nY 坐标矩阵：")
print(Y)

# 绘制网格点
plt.scatter(X, Y, color='red')
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Meshgrid Points")
plt.grid()
plt.show()
