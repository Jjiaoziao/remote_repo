import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def holographic_3d_localization():
    """
    水下点声源三维定位（基于测量面上声压幅值分析）
    输出：全息面和测量面上最大声压点的三维坐标，以及声源 z 坐标的逆推结果，并可视化验证
    """
    # ============= 1. 参数定义 =============
    m0 = 1000.0  # 介质密度 (kg/m^3)
    c0 = 1500.0  # 声速 (m/s)
    v0 = 0.01  # 振动速度幅值 (m/s)

    f = 1000.0  # 频率 (Hz)
    r0 = 0.1  # 点声源等效半径 (m)
    x0 = 10.0  # 声源真实 x 坐标 (m)
    y0 = 20.0  # 声源真实 y 坐标 (m)
    z0 = 0.0  # 声源真实 z 坐标 (m)

    # 用户输入全息面在 z 方向的坐标（参考面）
    try:
        zh = float(input("请输入全息面在z方向的坐标（单位：m）："))
    except Exception as e:
        print("输入有误，使用默认全息面 z=10.0 m")
        zh = 10.0

    # 用户输入测量面在 z 方向的坐标（重建面）
    try:
        zs_measure = float(input("请输入测量面在z方向的坐标（单位：m）："))
    except Exception as e:
        print("输入有误，使用默认测量面 z=8.0 m")
        zs_measure = 8.0

    # ============= 2. 生成网格 =============
    d = 1.0  # 网格间距 (m)
    Nx = 100  # x 方向网格数
    Ny = 200  # y 方向网格数

    # 生成以 (0,0) 为中心的二维坐标网格
    xl1 = (np.arange(Nx) - Nx / 2) * d  # x 坐标
    yl2 = (np.arange(Ny) - Ny / 2) * d  # y 坐标
    Xl1, Yl2 = np.meshgrid(xl1, yl2)

    # ============= 3. 全息面理论声压计算 =============
    w = 2 * np.pi * f  # 角频率
    k = w / c0  # 波数
    Q0 = 4 * np.pi * r0 ** 2 * v0  # 点声源强度

    # 计算全息面上各点到声源的距离
    rh = np.sqrt((Xl1 - x0) ** 2 + (Yl2 - y0) ** 2 + (zh - z0) ** 2)
    # 计算全息面声压（基于点声源模型）
    ph = 1j * m0 * c0 * k * Q0 * np.exp(1j * k * rh) / (4 * np.pi * rh)

    # 记录全息面上的最大声压值及其对应点的三维坐标
    ph_abs = np.abs(ph)
    max_ph = np.max(ph_abs)
    idx_max_holo = np.unravel_index(np.argmax(ph_abs), ph_abs.shape)
    holo_max_x = xl1[idx_max_holo[1]]
    holo_max_y = yl2[idx_max_holo[0]]
    holo_max_coord = (holo_max_x, holo_max_y, zh)

    # ============= 4. 利用角谱法将全息面声压传播到测量面 =============
    # 计算网格尺寸
    Lx = Nx * d
    Ly = Ny * d
    # 构造波数域采样
    n1 = np.fft.fftfreq(Nx, d=d) * Nx
    n2 = np.fft.fftfreq(Ny, d=d) * Ny
    kx = n1 * 2 * np.pi / Lx
    ky = n2 * 2 * np.pi / Ly
    Kx, Ky = np.meshgrid(kx, ky)

    # 计算 kz（对传播波与消逝波分别处理）
    kz = np.zeros_like(Kx, dtype=np.complex128)
    mask_propagating = (k ** 2 > Kx ** 2 + Ky ** 2)
    mask_evanescent = ~mask_propagating
    kz[mask_propagating] = np.sqrt(k ** 2 - (Kx[mask_propagating] ** 2 + Ky[mask_propagating] ** 2))
    kz[mask_evanescent] = 1j * np.sqrt((Kx[mask_evanescent] ** 2 + Ky[mask_evanescent] ** 2) - k ** 2)

    # 传递函数 Gd：由全息面 (z=zh) 传播到测量面 (z=zs_measure)
    propagation_distance = zh - zs_measure
    Gd = np.zeros_like(Kx, dtype=np.complex128)
    Gd[mask_propagating] = np.exp(1j * kz[mask_propagating] * propagation_distance)
    Gd[mask_evanescent] = np.exp(-np.abs(kz[mask_evanescent]) * propagation_distance)

    # 利用角谱法进行传播计算
    p_kh = np.fft.fftshift(np.fft.fft2(ph))
    p_ks = p_kh / Gd
    p_ss = np.fft.ifft2(np.fft.ifftshift(p_ks))
    p_ss_abs = np.abs(p_ss)

    # 找出测量面上最大声压值及其对应点的三维坐标
    max_pressure = np.max(p_ss_abs)
    idx_max_measure = np.unravel_index(np.argmax(p_ss_abs), p_ss_abs.shape)
    meas_max_x = xl1[idx_max_measure[1]]
    meas_max_y = yl2[idx_max_measure[0]]
    meas_max_coord = (meas_max_x, meas_max_y, zs_measure)

    # # 利用全息面和测量面的最大声压值计算比值
    # pressure_ratio = max_pressure / max_ph
    #
    # # 根据几何关系，利用全息面到声源的距离 zh 和该比值计算 r_h
    # r_h = zh / (1 - pressure_ratio)  # 由 r_s = r_h - Δz 推导
    #
    # # 由 r_h 和全息面到声源的距离关系，逆推声源的 z 坐标
    # estimated_z0 = zh - r_h  # 声源深度估计

    alpha = max_pressure / max_ph  # 即 |p_measure|/|p_holo|
    estimated_z0 = (alpha * zs_measure - zh) / (alpha - 1)

    # ============= 5. 打印定位结果 =============
    print("\n=== 全息面最大声压点三维坐标 ===")
    print(f"全息面最大声压点坐标: ({holo_max_coord[0]:.2f}, {holo_max_coord[1]:.2f}, {holo_max_coord[2]:.2f}) m")
    print("\n=== 测量面最大声压点三维坐标 ===")
    print(f"测量面最大声压点坐标: ({meas_max_coord[0]:.2f}, {meas_max_coord[1]:.2f}, {meas_max_coord[2]:.2f}) m")
    print("\n=== 声源定位逆推结果 ===")
    print(f"全息面 z 坐标: {zh:.2f} m")
    print(f"测量面 z 坐标: {zs_measure:.2f} m")
    print(f"全息面最大声压值: {max_ph:.4e} Pa")
    print(f"测量面最大声压值: {max_pressure:.4e} Pa")
    print(f"估计声源 z 坐标: {estimated_z0:.4f} m")

    # ============= 6. 可视化结果 =============
    plt.figure(figsize=(10, 4))
    # 2D 等高线图显示测量面声压分布
    plt.subplot(121)
    cp = plt.contourf(Xl1, Yl2, p_ss_abs, 50, cmap='viridis')
    plt.colorbar(cp, label='声压 (Pa)')
    plt.scatter(meas_max_coord[0], meas_max_coord[1], color='r', marker='x', s=100, label='测量面最大声压点')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title(f'测量面 z = {zs_measure:.2f} m 的声压分布')
    plt.legend()
    plt.grid(True)

    # 3D 图显示测量面声压分布
    ax = plt.subplot(122, projection='3d')
    ax.plot_surface(Xl1, Yl2, p_ss_abs, cmap='viridis', alpha=0.8)
    ax.scatter(meas_max_coord[0], meas_max_coord[1], max_pressure, color='r', s=100, label='测量面最大声压点')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('声压 (Pa)')
    ax.set_title('测量面声压三维分布')
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    holographic_3d_localization()
