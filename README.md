import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, dct

# 中文显示配置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ---------------------- 1. 生成测试信号 ----------------------
N = 8
x = np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.float64)
print(f"原始信号长度: {N}")
print(f"原始信号: {x}")

# ---------------------- 2. 延拓方式对比 ----------------------
# DFT周期延拓
x_dft_extend = np.tile(x, 2)
# DCT偶对称延拓
x_dct_extend = np.concatenate([x, x[::-1]])

# 绘制延拓图
plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.stem(np.arange(N), x, basefmt=' ', linefmt='b-', markerfmt='bo')
plt.title('原始信号', fontsize=12)
plt.xlabel('样本点n', fontsize=10)
plt.ylabel('幅值x[n]', fontsize=10)
plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 2)
plt.stem(np.arange(2*N), x_dft_extend, basefmt=' ', linefmt='r-', markerfmt='ro')
plt.axvline(x=N-0.5, color='k', linestyle='--', alpha=0.7, label='边界（跳变）')
plt.title('DFT隐含周期延拓', fontsize=12)
plt.xlabel('延拓后样本点n', fontsize=10)
plt.ylabel('幅值x[n]', fontsize=10)
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 3)
plt.stem(np.arange(2*N), x_dct_extend, basefmt=' ', linefmt='g-', markerfmt='go')
plt.axvline(x=N-0.5, color='k', linestyle='--', alpha=0.7, label='边界（连续）')
plt.title('DCT隐含偶对称延拓', fontsize=12)
plt.xlabel('延拓后样本点n', fontsize=10)
plt.ylabel('幅值x[n]', fontsize=10)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('延拓方式对比.png', dpi=300, bbox_inches='tight')
plt.show()

# ---------------------- 3. DFT与DCT系数计算 ----------------------
# DFT
X_dft = fft(x)
mag_dft = np.abs(X_dft)
energy_dft_total = np.sum(mag_dft**2)

# DCT-II（正交归一化）
X_dct = dct(x, norm='ortho')
mag_dct = np.abs(X_dct)
energy_dct_total = np.sum(mag_dct**2)

# 帕塞瓦尔验证
print(f"\nDFT总能量: {energy_dft_total:.2f}, 时域总能量: {np.sum(x**2):.2f}")
print(f"DCT总能量: {energy_dct_total:.2f}, 时域总能量: {np.sum(x**2):.2f}")

# 能量占比计算
k_list = np.arange(1, N+1)
energy_ratio_dft = np.array([np.sum(mag_dft[:k]**2) / energy_dft_total for k in k_list])
energy_ratio_dct = np.array([np.sum(mag_dct[:k]**2) / energy_dct_total for k in k_list])

# 绘制频谱对比
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.stem(np.arange(N), mag_dft, basefmt=' ', linefmt='r-', markerfmt='ro')
plt.title('DFT幅度谱', fontsize=12)
plt.xlabel('频率k', fontsize=10)
plt.ylabel('幅度|X[k]|', fontsize=10)
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.stem(np.arange(N), mag_dct, basefmt=' ', linefmt='g-', markerfmt='go')
plt.title('DCT-II幅度谱', fontsize=12)
plt.xlabel('频率k', fontsize=10)
plt.ylabel('幅度|X[k]|', fontsize=10)
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(k_list, energy_ratio_dft, 'r-o', label='DFT', linewidth=2)
plt.plot(k_list, energy_ratio_dct, 'g-s', label='DCT-II', linewidth=2)
plt.title('前k个系数能量占比对比', fontsize=12)
plt.xlabel('系数个数k', fontsize=10)
plt.ylabel('能量占比', fontsize=10)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(k_list)

plt.tight_layout()
plt.savefig('频谱与能量对比.png', dpi=300, bbox_inches='tight')
plt.show()

# ---------------------- 4. 边界与能量分析 ----------------------
dft_boundary_jump = np.abs(x[-1] - x[0])
print(f"\nDFT延拓边界跳变幅度: {dft_boundary_jump}")
print(f"DCT延拓边界跳变幅度: 0")

print(f"\n=== 能量集中性分析 ===")
print(f"前2个系数能量占比: DFT={energy_ratio_dft[1]:.2%}, DCT={energy_ratio_dct[1]:.2%}")
print(f"前4个系数能量占比: DFT={energy_ratio_dft[3]:.2%}, DCT={energy_ratio_dct[3]:.2%}")
print(f"\n结论: DCT通过偶对称延拓消除了边界跳变，能量高度集中在低频，因此更适合图像压缩")
