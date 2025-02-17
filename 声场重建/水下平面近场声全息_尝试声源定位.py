import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def holographic_3d_localization():
    """
    水下点声源三维定位（基于多重建面和声压幅度分析）
    输出：声源的三维坐标估计，并可视化验证
    """
    # ============= 1. 参数定义 =============
    m0 = 1000.0  # 介质密度 (kg/m^3)
    c0 = 1500.0  # 声速 (m/s)
    v0 = 0.01  # 振动速度幅值 (m/s)

    f = 1000.0  # 频率 (Hz)
    r0 = 0.1  # 点声源等效半径 (m)
    x0 = 0.0  # 声源真实 x 坐标 (m)
    y0 = 0.0  # 声源真实 y 坐标 (m)
    z0 = 0.0  # 声源真实 z 坐标 (m)
    zh = 10.0  # 全息面深度 (m)

    # 全息面网格参数
    d = 1.0  # 网格间距 (m)
    Nx = 100  # x方向网格数
    Ny = 100  # y方向网格数

    # ============= 2. 生成全息面网格 =============
    xl1 = (np.arange(Nx) - Nx / 2) * d
    yl2 = (np.arange(Ny) - Ny / 2) * d
    Xl1, Yl2 = np.meshgrid(xl1, yl2)

    # ============= 3. 全息面理论声压计算 =============
    w = 2 * np.pi * f  # 角频率
    k = w / c0  # 波数
    Q0 = 4 * np.pi * r0 ** 2 * v0  # 点声源强度

    # 全息面声压 (基于点声源模型)
    rh = np.sqrt((Xl1 - x0) ** 2 + (Yl2 - y0) ** 2 + (zh - z0) ** 2)
    ph = 1j * m0 * c0 * k * Q0 * np.exp(1j * k * rh) / (4 * np.pi * rh)

    # ============= 4. 多重建面深度扫描 =============
    print("开始深度扫描...")
    zs_list = np.linspace(-3.0, 3.0, 50)  # 扫描深度范围 [-3m, 3m]
    max_pressures = []
    max_positions = []  # 存储每个zs对应的最大声压点坐标 (x, y)

    for zs in zs_list:
        # --------- 4.1 角谱法反推至当前深度zs ---------
        # 波数分解
        Lx = Nx * d
        Ly = Ny * d
        n1 = np.fft.fftfreq(Nx, d=d) * Nx
        n2 = np.fft.fftfreq(Ny, d=d) * Ny
        kx = n1 * 2 * np.pi / Lx
        ky = n2 * 2 * np.pi / Ly
        Kx, Ky = np.meshgrid(kx, ky)

        # 计算kz (传播波与消逝波分开处理)
        kz = np.zeros_like(Kx, dtype=np.complex128)
        mask_propagating = (k ** 2 > Kx ** 2 + Ky ** 2)
        mask_evanescent = ~mask_propagating
        kz[mask_propagating] = np.sqrt(k ** 2 - (Kx[mask_propagating] ** 2 + Ky[mask_propagating] ** 2))
        kz[mask_evanescent] = 1j * np.sqrt((Kx[mask_evanescent] ** 2 + Ky[mask_evanescent] ** 2) - k ** 2)

        # 传递函数Gd
        Gd = np.zeros_like(Kx, dtype=np.complex128)
        Gd[mask_propagating] = np.exp(1j * kz[mask_propagating] * (zh - zs))
        Gd[mask_evanescent] = np.exp(-np.abs(kz[mask_evanescent]) * (zh - zs))  # 固定符号问题

        # 反推声压
        p_kh = np.fft.fftshift(np.fft.fft2(ph))  # FFT到波数域
        p_ks = p_kh / Gd
        p_ss = np.fft.ifft2(np.fft.ifftshift(p_ks))  # 反变换到空间域

        # --------- 4.2 记录最大声压值及其坐标 ---------
        p_abs = np.abs(p_ss)
        max_val = np.max(p_abs)
        idx_max = np.unravel_index(np.argmax(p_abs), p_abs.shape)

        max_x = xl1[idx_max[1]]  # 注意网格索引的对应关系
        max_y = yl2[idx_max[0]]

        max_pressures.append(max_val)
        max_positions.append((max_x, max_y))

    # ============= 5. 三维定位分析 =============
    # 寻找最大声压的深度
    idx_z_peak = np.argmax(max_pressures)
    estimated_z = zs_list[idx_z_peak]
    estimated_x, estimated_y = max_positions[idx_z_peak]

    print("\n=== 三维定位结果 ===")
    print(f"理论声源位置: ({x0:.2f}, {y0:.2f}, {z0:.2f}) m")
    print(f"估计声源位置: ({estimated_x:.2f}, {estimated_y:.2f}, {estimated_z:.2f}) m")

    # ============= 6. 可视化验证 =============
    # 6.1 最大声压随深度变化曲线
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.plot(zs_list, max_pressures, 'b-o', markersize=4)
    plt.axvline(z0, color='r', linestyle='--', label='真实深度')
    plt.xlabel('重建深度 z (m)')
    plt.ylabel('最大声压 (Pa)')
    plt.title('最大声压随重建深度变化')
    plt.grid(True)
    plt.legend()

    # 6.2 最优重建面声压分布
    fig = plt.figure(figsize=(10, 4))
    # 创建 3D 轴
    ax = fig.add_subplot(122, projection='3d')
    zs = estimated_z
    rs = np.sqrt((Xl1 - x0) ** 2 + (Yl2 - y0) ** 2 + (zs - z0) ** 2)
    ps = np.abs(1j * m0 * c0 * k * Q0 * np.exp(1j * k * rs) / (4 * np.pi * rs))
    # 绘制 3D 声压分布
    ax.plot_surface(Xl1, Yl2, ps, cmap='viridis', alpha=0.6)
    # 标注估计位置
    ax.scatter(estimated_x, estimated_y, np.max(ps), c='r', s=100, label='估计位置')
    # 设置标题和坐标轴标签
    ax.set_title(f'重建面 z={zs:.2f}m 的声压分布')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('声压 (Pa)')
    # 添加图例
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    holographic_3d_localization()
