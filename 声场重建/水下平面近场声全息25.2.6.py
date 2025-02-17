import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def holographic_simulation():
    """
    Python 形式的点声源平面近场声全息仿真，
    适用于水下环境，并在关键步骤加入详细注释。
    """

    # ============= (1) 介质与声源基础参数 =============
    # 修改部分：将介质从空气改为水
    m0 = 1000.0  # 水密度 (kg/m^3)
    c0 = 1500.0  # 声速 (m/s)  # 修改部分：水中的声速
    v0 = 2.5     # 振动速度, 用于计算源强度 Q0

    f  = 1000.0  # 频率 (Hz)
    r0 = 0.001   # 点声源等效半径 (m)
    x0 = 0.0     # 声源 x 坐标 (如需偏移源, 可改此值)
    y0 = 0.0     # 声源 y 坐标
    z0 = 0.0     # 声源 z 坐标
    w  = 2.0 * np.pi * f  # 角频率 w = 2πf
    k  = w / c0           # 波数 k = w/c

    # 点声源强度, Q0 = 4π r0^2 v0 (球面振动辐射假设)
    Q0 = 4.0 * np.pi * (r0**2) * v0

    # ============= (2) 全息面参数 =============
    zh = 0.10  # 全息面到声源面的距离 (m), 即 z=zh
    d  = 0.05  # 网格间距 (m)
    Nx = 20    # 全息面 x 方向网格数
    Ny = 20    # 全息面 y 方向网格数

    # 全息面物理孔径, 仅作记录/计算
    Lx = Nx * d
    Ly = Ny * d

    # 网格的索引 (0..Nx-1, 0..Ny-1)
    l1 = np.arange(Nx)
    l2 = np.arange(Ny)

    # 将网格中心对齐 0,0;
    # 例: Nx=20 => l1-9.5 => [-9.5..+9.5], 再乘以 d => x 坐标
    xl1 = (l1 - (Nx - 1)/2.0) * d
    yl2 = (l2 - (Ny - 1)/2.0) * d

    # ============= (3) 重建面参数 =============
    zs = 0.05  # 重建面到声源面的距离 (m), 即 z=zs

    # 这里仅复制一遍, 在 MATLAB 里是 xn1=xl1, yn2=yl2
    xn1 = xl1
    yn2 = yl2

    # ============= (4) 全息面 & 重建面理论声压 =============
    # 先构建网格点 (Xl1, Yl2)
    Xl1, Yl2 = np.meshgrid(xl1, yl2)  #将一维坐标数组转换为二维网格坐标矩阵

    # 计算从(0,0,0)到这两个平面的距离:
    #   rh => z=zh 全息面
    #   rs => z=zs 重建面
    # 如果需要源点偏移 (x0,y0,z0), 则:
    # rh = sqrt((Xl1-x0)^2 + (Yl2-y0)^2 + (zh-z0)^2)
    # 这里 x0,y0,z0=0, 简化为:
    rh = np.sqrt( (Xl1 - x0)**2 + (Yl2 - y0)**2 + (zh - z0)**2 )  #声源点到全息面每个网格点的距离
    rs = np.sqrt( (Xl1 - x0)**2 + (Yl2 - y0)**2 + (zs - z0)**2 )  #声源点到重建面每个网格点的距离

    # 点声源公式: p(r) = j·ρ·c·k·Q0 · exp(j·k·r) / (4πr)
    # Python 虚数单位用 1j
    ph = 1j * m0 * c0 * k * Q0 * np.exp(1j * k * rh) / (4.0 * np.pi * rh)  #全息面上的球面波声压
    ps = 1j * m0 * c0 * k * Q0 * np.exp(1j * k * rs) / (4.0 * np.pi * rs)  #重建面上的球面波声压

    # ============= (5) 可视化: 全息面 & 重建面(理论) =============
    fig1 = plt.figure(figsize=(10,4))

    ax1 = fig1.add_subplot(121, projection='3d')
    # plot_surface(X, Y, Z) - 其中 Z = abs(ph)
    ax1.plot_surface(Xl1, Yl2, np.abs(ph), cmap='viridis')
    ax1.set_title('全息面声压 |ph|')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')

    ax2 = fig1.add_subplot(122, projection='3d')
    ax2.plot_surface(Xl1, Yl2, np.abs(ps), cmap='viridis')
    ax2.set_title('重建面声压(理论) |ps|')
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)')

    plt.tight_layout()
    plt.show()

    # ============= (6) 角谱法反推到重建面 =============
    # 6.1 频率网格 n1, n2: [-0.5*(N-1) .. +0.5*(N-1)]
    n1 = np.arange(-0.5*(Nx-1), 0.5*(Nx-1) + 1) #二维傅里叶变换后的x方向索引
    n2 = np.arange(-0.5*(Ny-1), 0.5*(Ny-1) + 1)  #二维傅里叶变换后的y方向索引

    # kx, ky
    # 注意 MATLAB 里也对 ky 用 Lx 做除法, 如果严格要区分 x,y 可改 ky => / Ly
    kx = n1 * 2.0*np.pi / Lx  #物理波数
    ky = n2 * 2.0*np.pi / Ly  #物理波数

    Kx, Ky = np.meshgrid(kx, ky)  #将一维坐标数组转换为二维网格坐标矩阵

    # 6.2 判断传播波 / 倏逝波 (k^2 与 kx^2+ky^2 对比), 计算 kz
    # 以布尔掩码方式逐点处理
    kz = np.zeros_like(Kx, dtype=np.complex128)

    mask_propagating = (k**2 > (Kx**2 + Ky**2))  # 传播波区域：k² > Kx² + Ky²
    mask_evanescent = ~mask_propagating         # 倏逝波区域：k² ≤ Kx² + Ky²

    # 当 k^2 > kx^2 + ky^2,  k_z = sqrt(k^2 - (kx^2+ky^2))
    kz[mask_propagating] = np.sqrt( k**2 - (Kx[mask_propagating]**2 + Ky[mask_propagating]**2) )  #计算 z 方向的波数，计算实数部分，即传播波的贡献
    # 当 k^2 < kx^2 + ky^2,  k_z = i * sqrt((kx^2+ky^2) - k**2)
    kz[mask_evanescent]  = 1j * np.sqrt( (Kx[mask_evanescent]**2 + Ky[mask_evanescent]**2) - k**2 )  #计算 z 方向的波数，计算虚数部分，即倏逝波的贡献

    # 6.3 构造传播算子 Gd:类似于声传播的衰减因子
    # if k^2>kx^2+ky^2 => Gd=exp(i*kz*(zh-zs))
    # else => Gd=exp(-kz*(zh-zs))
    Gd = np.zeros_like(Kx, dtype=np.complex128)
    Gd[mask_propagating] = np.exp(1j * kz[mask_propagating] * (zh - zs))
    Gd[mask_evanescent]  = np.exp(-kz[mask_evanescent] * (zh - zs))

    # 6.4 做 FFT2 + fftshift => pkhh
    pkh = np.fft.fft2(ph)  # 对全息面进行2D FFT
    pkhh = np.fft.fftshift(pkh)  # 对结果进行fftshift，将零频移到数组中心，以确保频域数据的对称性和直观性

    # 逆向传播: pks = pkhh / Gd
    pks = pkhh / Gd  # 从全息面逆向传播到重建面

    # 6.5 ifft2 => 重建面反推结果 pss
    pss = np.fft.ifft2(pks)  # 对结果进行逆向FFT，从频域数据恢复到空间域数据

    # ============= (7) 可视化 角谱法得到的重建面声压 =============
    fig2 = plt.figure()
    ax3 = fig2.add_subplot(111, projection='3d')
    ax3.plot_surface(Xl1, Yl2, np.abs(pss), cmap='viridis')
    ax3.set_title('重建面声压(反推) |pss|')
    ax3.set_xlabel('x (m)')
    ax3.set_ylabel('y (m)')

    plt.show()


# 如果想直接运行脚本:
if __name__ == '__main__':
    holographic_simulation()

