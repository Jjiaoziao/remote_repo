import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def holographic_3d_localization():
    """
    主要过程：
      1) 定义介质和声源参数：包括密度、声速、频率、点源强度等；
      2) 用户输入：全息面与重建面坐标（z方向），同时沿 x 方向平面也使用相同距离；
      3) 沿 z 方向的 NAH：在 (x,y) 上做测量面和重建面；用角谱法在频域传播算子求解；
      4) 沿 x 方向的 NAH：在 (y,z) 上做测量面和重建面；用角谱法在频域传播算子求解；
      5) 分别找到最大声压点位置，并可视化(2D等高线 + 3D表面图)。
    """

    # ============= 1. 基础物理与声源参数 =============
    m0 = 1000.0  # 水密度 (kg/m^3)
    c0 = 1500.0  # 声速 (m/s)
    v0 = 0.01    # 振动速度幅值 (m/s)，用以计算点源强度
    f  = 1000.0  # 频率 (Hz)
    r0 = 0.1     # 点声源等效半径 (m)

    # 声源坐标 (示例：将点声源放在 (10,10,20))
    x0_true = 10.0
    y0_true = 10.0
    z0_true = 20.0

    # 角频率 & 波数 & 点声源强度
    w  = 2 * np.pi * f       # 角频率 ω = 2πf
    k  = w / c0              # 波数 k = ω/c
    Q0 = 4 * np.pi * (r0**2) * v0  # 点声源强度 Q0 = 4πr0²·v0

    # ============= 2. 获取用户输入: 全息面 & 重建面 (z方向) =============
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

    # 为了示例对称性，在 x 方向也使用同样的数值
    xHolographic = zHolographic
    xReconstruction = zReconstruction

    # ============= 3. 沿 Z 方向(只处理 x,y 面) =============
    # d_xy: (x,y)方向网格间距； Nx, Ny: 网格点数量
    d_xy = 1.0
    Nx = 100
    Ny = 200

    # 构造 x、y 坐标，使得中心对齐 (0,0)
    xl = (np.arange(Nx) - Nx / 2) * d_xy  # x坐标数组
    yl = (np.arange(Ny) - Ny / 2) * d_xy  # y坐标数组
    # 网格化 => (Ny, Nx) 形状
    X, Y = np.meshgrid(xl, yl)

    # (1) 计算在全息面 z=zHolographic 处的声压 (球面波公式)
    # 距离 r_h = sqrt((x - x0)^2 + (y - y0)^2 + (zH - z0)^2)
    r_h = np.sqrt((X - x0_true)**2 + (Y - y0_true)**2 + (zHolographic - z0_true)**2)
    # 复数声压: p = jρc k Q0 * exp(jkr)/(4πr)
    ph = (1j*m0*c0*k*Q0 * np.exp(1j*k*r_h) / (4*np.pi*r_h))
    ph_abs = np.abs(ph)         # 幅值
    max_ph = np.max(ph_abs)     # 该面上的最大值

    # 找到该面上最大值位置的索引 => (行,列)
    idx_max_holo = np.unravel_index(np.argmax(ph_abs), ph_abs.shape)
    # 由索引映射回坐标， idx_max_holo[1] 对应 x, idx_max_holo[0] 对应 y
    holo_max_x = xl[idx_max_holo[1]]
    holo_max_y = yl[idx_max_holo[0]]

    # (2) 使用角谱法，从全息面 z=zHolographic 传播到 z=zReconstruction
    #     先构造频域网格 (kx, ky)，再做 FFT + 乘传播算子 + IFFT

    Lx = Nx * d_xy  # x方向物理尺寸
    Ly = Ny * d_xy  # y方向物理尺寸

    # freqfreq(N, d=Δ) => [-0.5N, ..., 0.5N-1]/N，乘 Nx,Ny => 频谱取值
    n1 = np.fft.fftfreq(Nx, d=d_xy)*Nx
    n2 = np.fft.fftfreq(Ny, d=d_xy)*Ny
    kx = n1 * 2*np.pi / Lx  # x方向波数
    ky = n2 * 2*np.pi / Ly  # y方向波数
    KX, KY = np.meshgrid(kx, ky)  # shape=(Ny, Nx)

    # kz: z方向波数，区分传播波/倏逝波
    kz = np.zeros_like(KX, dtype=np.complex128) #创建一个和 KX 形状相同且元素全为 0 的数组,数组的数据类型为复数
    mask_propagating = (k**2 > (KX**2 + KY**2))  #该数组的元素为True时，对应的波数是传播波，否则为倏逝波
    # 对于 k^2>kx^2+ky^2 => kz= sqrt(k^2 - (kx^2+ky^2)) (实数)
    kz[mask_propagating] = np.sqrt(k**2 - (KX[mask_propagating]**2 + KY[mask_propagating]**2))  #为传播波时，计算 kz 的实数部分，即传播波的贡献
    mask_evanescent = ~mask_propagating #该数组的元素为True时，对应的波数是倏逝波，否则为传播波，~表示按位取反
    # 对于 k^2<kx^2+ky^2 => kz= i * sqrt((kx^2+ky^2) - k^2) (虚数)
    kz[mask_evanescent] = 1j*np.sqrt((KX[mask_evanescent]**2 + KY[mask_evanescent]**2) - k**2)  #为倏逝波时，计算 kz 的虚数部分，即倏逝波的贡献

    # 构造传播算子 Gd_z, distance=(zH - zR)
    propagation_distance_z = (zHolographic - zReconstruction)  #全息面和重建面之间的距离
    Gd_z = np.zeros_like(KX, dtype=np.complex128)  #初始化传播因子数组,创建一个和 KX 形状相同且元素全为 0 的数组,数组的数据类型为复数
    # 传播波的传播算子
    Gd_z[mask_propagating] = np.exp(1j * kz[mask_propagating] * propagation_distance_z)
    # 倏逝波的传播算子
    Gd_z[mask_evanescent] = np.exp(-np.abs(kz[mask_evanescent]) * propagation_distance_z)

    # 全息面声压做 2D FFT + fftshift => p_kh
    p_kh = np.fft.fftshift(np.fft.fft2(ph)) #将空间域的全息面声压转换至波数域,并对波数域的结果进行中心化处理
    # 乘(或除)传播算子 => 频域传播
    p_ks = p_kh / Gd_z  #得到重建面在波数域的声压
    # IFFT => 重建面声压 p_ss
    p_ss = np.fft.ifft2(np.fft.ifftshift(p_ks))  #将波数域的重建面的声压转换至空间域,得到重建面上的声压分布
    p_ss_abs = np.abs(p_ss)  #获取重建面的声压的幅值

    # 找该重建面上最大值及其 (x, y) 坐标
    max_pressure_zDir = np.max(p_ss_abs)  #重建面的声压最大值
    idx_max_measure_z = np.unravel_index(np.argmax(p_ss_abs), p_ss_abs.shape) #找出声压幅值最大值在数组 p_ss_abs 中的二维索引
    recon_max_x_z = xl[idx_max_measure_z[1]]  #由二维索引映射到x坐标
    recon_max_y_z = yl[idx_max_measure_z[0]]  #由二维索引映射到y坐标

    # ============= 4. 沿 X 方向(只处理 y,z 面) =============
    # 原理类似，只不过固定 x=xHolographic，网格是 (y, z)

    d_yz = 1.0
    N_y2 = 200
    N_z2 = 100

    y2 = (np.arange(N_y2) - N_y2/2)*d_yz
    z2 = (np.arange(N_z2) - N_z2/2)*d_yz
    Y2, Z2 = np.meshgrid(y2, z2)  # shape=(N_z2, N_y2)

    # 全息面 (x=xHolographic) 上声压
    # 距离 r_hx = sqrt( (xH - x0)^2 + (y - y0)^2 + (z - z0)^2 )
    r_hx = np.sqrt((xHolographic - x0_true)**2 + (Y2 - y0_true)**2 + (Z2 - z0_true)**2)
    # 避免 r=0
    r_hx = np.where(r_hx < 1e-9, 1e-9, r_hx)

    ph_x = (1j*m0*c0*k*Q0 * np.exp(1j*k*r_hx)/(4*np.pi*r_hx))
    ph_x_abs = np.abs(ph_x)
    max_ph_x = np.max(ph_x_abs)

    idx_max_holo_x = np.unravel_index(np.argmax(ph_x_abs), ph_x_abs.shape)
    holo_max_y_x = y2[idx_max_holo_x[1]]
    holo_max_z_x = z2[idx_max_holo_x[0]]

    # 角谱法传播到 重建面 (x=xReconstruction)
    Ly2 = N_y2 * d_yz
    Lz2 = N_z2 * d_yz
    n_y2 = np.fft.fftfreq(N_y2, d=d_yz)*N_y2
    n_z2 = np.fft.fftfreq(N_z2, d=d_yz)*N_z2
    ky2 = n_y2 * 2*np.pi / Ly2
    kz2 = n_z2 * 2*np.pi / Lz2
    KY2, KZ2 = np.meshgrid(ky2, kz2)

    kx2 = np.zeros_like(KY2, dtype=np.complex128)
    mask_prop_x = (k**2 > (KY2**2 + KZ2**2))
    mask_evan_x = ~mask_prop_x

    kx2[mask_prop_x] = np.sqrt(k**2 - (KY2[mask_prop_x]**2 + KZ2[mask_prop_x]**2))
    kx2[mask_evan_x] = 1j*np.sqrt((KY2[mask_evan_x]**2 + KZ2[mask_evan_x]**2) - k**2)

    propagation_distance_x = (xHolographic - xReconstruction)
    Gd_x = np.zeros_like(KY2, dtype=np.complex128)
    Gd_x[mask_prop_x] = np.exp(1j*kx2[mask_prop_x]*propagation_distance_x)
    Gd_x[mask_evan_x] = np.exp(-np.abs(kx2[mask_evan_x])*propagation_distance_x)

    p_kh_x = np.fft.fftshift(np.fft.fft2(ph_x))
    p_ks_x = p_kh_x / Gd_x
    p_ss_x = np.fft.ifft2(np.fft.ifftshift(p_ks_x))
    p_ss_x_abs = np.abs(p_ss_x)

    max_pressure_xDir = np.max(p_ss_x_abs)
    idx_max_measure_x = np.unravel_index(np.argmax(p_ss_x_abs), p_ss_x_abs.shape)
    recon_max_y_x = y2[idx_max_measure_x[1]]
    recon_max_z_x = z2[idx_max_measure_x[0]]

    # ============= 5. 打印结果信息 =============
    print("\n[沿 Z 方向] 全息面 (z = {:.2f}) 上最大声压点 (x, y) = ({:.2f}, {:.2f}), "
          "最大声压 = {:.4e} Pa".format(zHolographic, holo_max_x, holo_max_y, max_ph))
    print("           重建面 (z = {:.2f}) 上最大声压点 (x, y) = ({:.2f}, {:.2f}), "
          "最大声压 = {:.4e} Pa\n".format(zReconstruction, recon_max_x_z, recon_max_y_z, max_pressure_zDir))

    print("[沿 X 方向] 全息面 (x = {:.2f}) 上最大声压点 (y, z) = ({:.2f}, {:.2f}), "
          "最大声压 = {:.4e} Pa".format(xHolographic, holo_max_y_x, holo_max_z_x, max_ph_x))
    print("           重建面 (x = {:.2f}) 上最大声压点 (y, z) = ({:.2f}, {:.2f}), "
          "最大声压 = {:.4e} Pa\n".format(xReconstruction, recon_max_y_x, recon_max_z_x, max_pressure_xDir))

    # ============= 6. 可视化 - 2D contourf + 3D surface =============

    ## 6.1 沿 z 方向
    fig1 = plt.figure(figsize=(12, 5))

    # (a) 2D 等高线分布
    ax1_2d = fig1.add_subplot(1, 2, 1)
    cp1 = ax1_2d.contourf(X, Y, p_ss_abs, 50, cmap='viridis')
    # 绘制声压幅度等高线图
    plt.colorbar(cp1, ax=ax1_2d, label='声压 (Pa)')
    # 在图上标记出该面最大声压点
    ax1_2d.scatter(recon_max_x_z, recon_max_y_z, color='r', marker='x', s=100, label='重建面最大声压')
    ax1_2d.set_xlabel('x (m)')
    ax1_2d.set_ylabel('y (m)')
    ax1_2d.set_title(f'[沿 z] 重建面 z={zReconstruction:.2f} m')
    ax1_2d.legend()
    ax1_2d.grid(True)

    # (b) 3D surface 绘图 => plot_surface(X, Y, Z值)
    ax1_3d = fig1.add_subplot(1, 2, 2, projection='3d')
    ax1_3d.plot_surface(X, Y, p_ss_abs, cmap='viridis')
    ax1_3d.set_xlabel('x (m)')
    ax1_3d.set_ylabel('y (m)')
    ax1_3d.set_zlabel('声压幅度')
    ax1_3d.set_title('[沿 z] 重建面声压 (3D表面)')

    ## 6.2 沿 x 方向
    fig2 = plt.figure(figsize=(12, 5))

    # (a) 2D 等高线分布
    ax2_2d = fig2.add_subplot(1, 2, 1)
    cp2 = ax2_2d.contourf(Y2, Z2, p_ss_x_abs, 50, cmap='viridis')
    # 绘制 y-z 平面上的声压等高线图
    plt.colorbar(cp2, ax=ax2_2d, label='声压 (Pa)')
    ax2_2d.scatter(recon_max_y_x, recon_max_z_x, color='r', marker='x', s=100, label='重建面最大声压')
    ax2_2d.set_xlabel('y (m)')
    ax2_2d.set_ylabel('z (m)')
    ax2_2d.set_title(f'[沿 x] 重建面 x={xReconstruction:.2f} m')
    ax2_2d.legend()
    ax2_2d.grid(True)

    # (b) 3D surface 绘图 => plot_surface(Y2, Z2, 声压幅度)
    ax2_3d = fig2.add_subplot(1, 2, 2, projection='3d')
    ax2_3d.plot_surface(Y2, Z2, p_ss_x_abs, cmap='viridis')
    ax2_3d.set_xlabel('y (m)')
    ax2_3d.set_ylabel('z (m)')
    ax2_3d.set_zlabel('声压幅度')
    ax2_3d.set_title('[沿 x] 重建面声压 (3D表面)')

    # 调整布局, 显示图像
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    holographic_3d_localization()
