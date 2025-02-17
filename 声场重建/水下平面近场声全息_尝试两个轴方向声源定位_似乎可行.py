import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def holographic_3d_localization():
    """
    演示如何在水下环境下，分别在两组正交平面上（沿z方向的平面 & 沿x方向的平面）
    进行近场声全息(NAH)计算，并找到重建面上的最大声压点。

    注意：
      1) 我们将“沿 z 方向”平面理解为 z=常数（水平面），只分析 (x, y) 坐标；
      2) 将“沿 x 方向”平面理解为 x=常数（竖直面），只分析 (y, z) 坐标；
      3) 去除了原代码中基于“幅度比”反推 x、z 坐标的部分，仅保留“最大声压点”的查找流程。
      4) 因为示例中仅做演示，所以我们将同一个用户输入的数值复用到“z方向”和“x方向”计算：
         - zHolographic, zReconstruction：用户输入的两个坐标值，分别指“沿 z 方向时”的全息面和重建面
         - xHolographic, xReconstruction：这里简单地令 xHolographic = zHolographic, xReconstruction = zReconstruction，
           以示范“沿 x 方向”时如何做类似的计算。
    """

    # ============= 1. 基础物理与声源参数 =============
    m0 = 1000.0  # 介质(水)密度 (kg/m^3)
    c0 = 1500.0  # 声速 (m/s)
    v0 = 0.01  # 振动速度幅值 (m/s)
    f = 1000.0  # 频率 (Hz)
    r0 = 0.1  # 点声源等效半径 (m)

    # 声源“真实”坐标（用于仿真/对比，这里设为原点）
    x0_true = 10.0
    y0_true = 10.0
    z0_true = 20.0

    # 角频率 & 波数 & 点声源强度
    w = 2 * np.pi * f  # 角频率 ω = 2πf
    k = w / c0  # 波数 k = ω / c
    Q0 = 4 * np.pi * (r0 ** 2) * v0  # 点声源强度 Q0 = 4πr0²·v0

    # ============= 2. 获取用户输入的两个平面位置 =============
    # 说明：我们这里让用户输入“沿 z 方向的全息面坐标”和“沿 z 方向的重建面坐标”。
    #       同时，将它们复用到“沿 x 方向”的平面坐标上。
    try:
        zHolographic = float(input("请输入沿Z方向全息面(z坐标)（单位：m）: "))
    except Exception as e:
        print("输入有误，使用默认全息面 z=10.0 m")
        zHolographic = 10.0

    try:
        zReconstruction = float(input("请输入沿Z方向重建面(z坐标)（单位：m）: "))
    except Exception as e:
        print("输入有误，使用默认重建面 z=5.0 m")
        zReconstruction = 5.0

    # 这里为了演示“沿x方向平面”与“沿z方向平面”的对称性，做一个简单赋值：
    xHolographic = zHolographic
    xReconstruction = zReconstruction

    # ============= 3. 沿 Z 方向的计算：只处理 (x, y) 平面 =============
    # 3.1 定义网格参数
    d_xy = 1.0  # 网格间距 (m)，演示用
    Nx = 100  # x 方向网格数
    Ny = 200  # y 方向网格数

    # 生成 x、y 网格坐标，使其中心对齐 (0,0)
    xl = (np.arange(Nx) - Nx / 2) * d_xy
    yl = (np.arange(Ny) - Ny / 2) * d_xy
    X, Y = np.meshgrid(xl, yl)  # X, Y 的 shape = (Ny, Nx)

    # 3.2 计算 全息面(z=zHolographic) 上的声压分布 (理论球面波)
    r_h = np.sqrt((X - x0_true) ** 2 +
                  (Y - y0_true) ** 2 +
                  (zHolographic - z0_true) ** 2)
    ph = (1j * m0 * c0 * k * Q0 *
          np.exp(1j * k * r_h) / (4 * np.pi * r_h))
    ph_abs = np.abs(ph)
    max_ph = np.max(ph_abs)  # 全息面声压最大值

    # 找到全息面最大声压处的 (x, y) 索引
    idx_max_holo = np.unravel_index(np.argmax(ph_abs), ph_abs.shape)
    holo_max_x = xl[idx_max_holo[1]]  # 注意行、列与 x,y 对应
    holo_max_y = yl[idx_max_holo[0]]

    # 3.3 使用“角谱法”传播到重建面 (z=zReconstruction)
    #     构造频域网格 (kx, ky)，再做 FFT -> 乘传播算子 -> IFFT
    Lx = Nx * d_xy
    Ly = Ny * d_xy
    # np.fft.fftfreq(N, d=步长) 会返回 [-0.5N, ..., 0.5N-1]/N 形式的频率刻度
    n1 = np.fft.fftfreq(Nx, d=d_xy) * Nx
    n2 = np.fft.fftfreq(Ny, d=d_xy) * Ny
    kx = n1 * 2 * np.pi / Lx
    ky = n2 * 2 * np.pi / Ly
    KX, KY = np.meshgrid(kx, ky)  # 同样 shape = (Ny, Nx)

    # 区分传播波/消逝波
    kz = np.zeros_like(KX, dtype=np.complex128)
    mask_propagating = (k ** 2 > (KX ** 2 + KY ** 2))
    mask_evanescent = ~mask_propagating

    kz[mask_propagating] = np.sqrt(k ** 2 - (KX[mask_propagating] ** 2 + KY[mask_propagating] ** 2))
    kz[mask_evanescent] = 1j * np.sqrt((KX[mask_evanescent] ** 2 + KY[mask_evanescent] ** 2) - k ** 2)

    # 传播算子：从 zHolographic 到 zReconstruction
    propagation_distance_z = (zHolographic - zReconstruction)
    Gd_z = np.zeros_like(KX, dtype=np.complex128)
    Gd_z[mask_propagating] = np.exp(1j * kz[mask_propagating] * propagation_distance_z)
    Gd_z[mask_evanescent] = np.exp(-np.abs(kz[mask_evanescent]) * propagation_distance_z)

    # 频域处理： FFT -> 乘以传播算子 -> IFFT
    p_kh = np.fft.fftshift(np.fft.fft2(ph))  # 全息面声压的 2D FFT
    p_ks = p_kh / Gd_z  # 这里是“反向传播”或“正向传播”的处理
    p_ss = np.fft.ifft2(np.fft.ifftshift(p_ks))  # 得到重建面上的声压分布
    p_ss_abs = np.abs(p_ss)

    # 找到重建面最大声压点 (x, y)
    max_pressure_zDir = np.max(p_ss_abs)
    idx_max_measure_z = np.unravel_index(np.argmax(p_ss_abs), p_ss_abs.shape)
    recon_max_x_z = xl[idx_max_measure_z[1]]
    recon_max_y_z = yl[idx_max_measure_z[0]]

    # ============= 4. 沿 X 方向的计算：只处理 (y, z) 平面 =============
    #    类似地，把 xHolographic, xReconstruction 看成 x=常数 的两个平面
    # 4.1 定义网格参数 (y, z)
    d_yz = 1.0  # 网格间距 (m)
    N_y2 = 200  # y 方向网格数
    N_z2 = 100  # z 方向网格数

    y2 = (np.arange(N_y2) - N_y2 / 2) * d_yz
    z2 = (np.arange(N_z2) - N_z2 / 2) * d_yz
    Y2, Z2 = np.meshgrid(y2, z2)  # shape = (N_z2, N_y2)

    # 4.2 计算 全息面 (x = xHolographic) 上的声压
    r_hx = np.sqrt((xHolographic - x0_true) ** 2 +
                   (Y2 - y0_true) ** 2 +
                   (Z2 - z0_true) ** 2)

    # 避免除 0
    r_hx = np.where(r_hx < 1e-9, 1e-9, r_hx)

    ph_x = (1j * m0 * c0 * k * Q0 *
            np.exp(1j * k * r_hx) / (4 * np.pi * r_hx))
    ph_x_abs = np.abs(ph_x)
    max_ph_x = np.max(ph_x_abs)

    idx_max_holo_x = np.unravel_index(np.argmax(ph_x_abs), ph_x_abs.shape)
    holo_max_y_x = y2[idx_max_holo_x[1]]
    holo_max_z_x = z2[idx_max_holo_x[0]]

    # 4.3 角谱法传播到 重建面 (x = xReconstruction)
    Ly2 = N_y2 * d_yz
    Lz2 = N_z2 * d_yz
    n_y2 = np.fft.fftfreq(N_y2, d=d_yz) * N_y2
    n_z2 = np.fft.fftfreq(N_z2, d=d_yz) * N_z2
    ky2 = n_y2 * 2 * np.pi / Ly2
    kz2 = n_z2 * 2 * np.pi / Lz2
    KY2, KZ2 = np.meshgrid(ky2, kz2)  # shape = (N_z2, N_y2)

    # 计算 kx2
    kx2 = np.zeros_like(KY2, dtype=np.complex128)
    mask_prop_x = (k ** 2 > KY2 ** 2 + KZ2 ** 2)
    mask_evan_x = ~mask_prop_x

    kx2[mask_prop_x] = np.sqrt(k ** 2 - (KY2[mask_prop_x] ** 2 + KZ2[mask_prop_x] ** 2))
    kx2[mask_evan_x] = 1j * np.sqrt((KY2[mask_evan_x] ** 2 + KZ2[mask_evan_x] ** 2) - k ** 2)

    propagation_distance_x = (xHolographic - xReconstruction)
    Gd_x = np.zeros_like(KY2, dtype=np.complex128)
    Gd_x[mask_prop_x] = np.exp(1j * kx2[mask_prop_x] * propagation_distance_x)
    Gd_x[mask_evan_x] = np.exp(-np.abs(kx2[mask_evan_x]) * propagation_distance_x)

    p_kh_x = np.fft.fftshift(np.fft.fft2(ph_x))
    p_ks_x = p_kh_x / Gd_x
    p_ss_x = np.fft.ifft2(np.fft.ifftshift(p_ks_x))
    p_ss_x_abs = np.abs(p_ss_x)

    max_pressure_xDir = np.max(p_ss_x_abs)
    idx_max_measure_x = np.unravel_index(np.argmax(p_ss_x_abs), p_ss_x_abs.shape)
    recon_max_y_x = y2[idx_max_measure_x[1]]
    recon_max_z_x = z2[idx_max_measure_x[0]]

    # ============= 5. 打印结果：全息面 & 重建面 最大声压点信息 =============
    print("\n[沿 Z 方向] 全息面 (z = {:.2f}) 上最大声压点 (x, y) = ({:.2f}, {:.2f}), "
          "最大声压 = {:.4e} Pa".format(zHolographic, holo_max_x, holo_max_y, max_ph))
    print("           重建面 (z = {:.2f}) 上最大声压点 (x, y) = ({:.2f}, {:.2f}), "
          "最大声压 = {:.4e} Pa\n".format(zReconstruction, recon_max_x_z, recon_max_y_z, max_pressure_zDir))

    print("[沿 X 方向] 全息面 (x = {:.2f}) 上最大声压点 (y, z) = ({:.2f}, {:.2f}), "
          "最大声压 = {:.4e} Pa".format(xHolographic, holo_max_y_x, holo_max_z_x, max_ph_x))
    print("           重建面 (x = {:.2f}) 上最大声压点 (y, z) = ({:.2f}, {:.2f}), "
          "最大声压 = {:.4e} Pa\n".format(xReconstruction, recon_max_y_x, recon_max_z_x, max_pressure_xDir))

    print("=== 提示 ===")
    print("以上仅演示沿 z、x 两个正交方向的平面各自做 NAH 并找最大声压点。\n"
          "通常若要完整三维定位，会结合更多测量面或三维阵列进行综合反演。\n"
          "此示例主要展示代码流程与命名规范。")

    # ============= 6. 可视化：沿 z 方向、x 方向重建面的声压分布 =============
    plt.figure(figsize=(12, 5))

    # (1) 沿 z 方向重建面 (x,y) 声压分布
    plt.subplot(1, 2, 1)
    cp1 = plt.contourf(X, Y, p_ss_abs, 50, cmap='viridis')
    plt.colorbar(cp1, label='声压 (Pa)')
    plt.scatter(recon_max_x_z, recon_max_y_z, color='r', marker='x', s=100, label='重建面最大声压点')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title(f'[沿 z] 重建面 z={zReconstruction:.2f} m')
    plt.legend()
    plt.grid(True)

    # (2) 沿 x 方向重建面 (y,z) 声压分布
    plt.subplot(1, 2, 2)
    cp2 = plt.contourf(Y2, Z2, p_ss_x_abs, 50, cmap='viridis')
    plt.colorbar(cp2, label='声压 (Pa)')
    plt.scatter(recon_max_y_x, recon_max_z_x, color='r', marker='x', s=100, label='重建面最大声压点')
    plt.xlabel('y (m)')
    plt.ylabel('z (m)')
    plt.title(f'[沿 x] 重建面 x={xReconstruction:.2f} m')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    holographic_3d_localization()
