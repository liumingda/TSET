# import torch
# import matplotlib.pyplot as plt
# import numpy as np

# class EDMModel:
#     def __init__(self, sigma_min=0.002, sigma_data=0.5):
#         self.sigma_min = sigma_min
#         self.sigma_data = sigma_data

#     def denoise_fn(self, x, c_noise, cond):
#         # 让 c_noise 扩展到与 x 的维度匹配 [B, 1, T]
#         c_noise = c_noise.view(-1, 1, 1)  # 将 c_noise 变成 [B, 1, 1]
#         c_noise = c_noise.expand(-1, 1, x.shape[2])  # 扩展到 [B, 1, T]
        
#         # 模拟去噪过程：根据噪声调整输入
#         return x * (1 - c_noise)  # 这里假设去噪是简单的加权操作

#     def EDMPrecond(self, x, sigma, cond):
#         sigma = sigma.reshape(-1, 1, 1)
        
#         # 计算预条件化系数
#         c_skip = self.sigma_data ** 2 / ((sigma - self.sigma_min) ** 2 + self.sigma_data ** 2)
#         c_out = (sigma - self.sigma_min) * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
#         c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
#         c_noise = sigma.log() / 4
        
#         # 调用去噪函数进行处理
#         F_x = self.denoise_fn(c_in * x, c_noise.flatten(), cond)
        
#         # 计算去噪后的输出数据
#         D_x = c_skip * x + c_out * (F_x.squeeze(1))
        
#         return D_x

# # 创建模型实例
# model = EDMModel()

# # 模拟一些数据
# B, T = 10, 100  # Batch size 和 Time steps
# x = torch.randn(B, 1, T)  # 输入数据（随机生成）

# # 模拟噪声强度 sigma
# sigma = torch.rand(B) * 0.1 + 0.01  # 噪声强度范围在 [0.01, 0.1] 之间

# # 条件信息 cond，假设我们用全0的矩阵作为条件
# cond = torch.zeros(B, 256, T)

# # 计算预条件化后的数据
# D_x = model.EDMPrecond(x, sigma, cond)

# # 可视化原始数据和预处理后的数据
# plt.figure(figsize=(10, 5))

# # 原始数据
# plt.subplot(1, 2, 1)
# plt.plot(x[0].numpy().flatten(), label='Original x')
# plt.title("Original Input (x)")
# plt.xlabel("Time Step")
# plt.ylabel("Amplitude")
# plt.grid(True)

# # 预条件化后的数据
# plt.subplot(1, 2, 2)
# plt.plot(D_x[0].numpy().flatten(), label='Processed D_x', color='r')
# plt.title("Processed Output (D_x)")
# plt.xlabel("Time Step")
# plt.ylabel("Amplitude")
# plt.grid(True)

# plt.tight_layout()
# plt.show()


import torch
import matplotlib.pyplot as plt
import numpy as np

# 假设的 sigma 范围和参数
sigma_min = 0.002
sigma_data = 0.5
sigma = torch.linspace(0.01, 0.1, steps=100)  # sigma 从 0.01 到 0.1，生成 100 个样本点

# 计算 c_skip, c_out, c_in 和 c_noise
c_skip = sigma_data ** 2 / ((sigma - sigma_min) ** 2 + sigma_data ** 2)
c_out = (sigma - sigma_min) * sigma_data / (sigma ** 2 + sigma_data ** 2).sqrt()
c_in = 1 / (sigma_data ** 2 + sigma ** 2).sqrt()
c_noise = sigma.log() / 4

# 绘制这些系数
plt.figure(figsize=(10, 6))

plt.plot(sigma.numpy(), c_skip.numpy(), label='c_skip', color='b')
plt.plot(sigma.numpy(), c_out.numpy(), label='c_out', color='r')
plt.plot(sigma.numpy(), c_in.numpy(), label='c_in', color='g')
plt.plot(sigma.numpy(), c_noise.numpy(), label='c_noise', color='m')

# 添加标题和标签
plt.title("Coefficients vs. Sigma")
plt.xlabel("Sigma")
plt.ylabel("Coefficient Value")
plt.legend()

# 显示图形
plt.grid(True)
plt.show()
