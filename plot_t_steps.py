import torch
import matplotlib.pyplot as plt

# 定义一些参数
sigma_min = 0.002
sigma_max = 80
rho = 7
N = 25

# 生成扩散步长
step_indices = torch.arange(N)  # [0, 1, 2, ..., 24]
t_steps = (sigma_min ** (1 / rho) + step_indices / (N - 1) * (sigma_max ** (1 / rho) - sigma_min ** (1 / rho))) ** rho

# 对 t_steps 进行四舍五入处理（假设 round_sigma 是对噪声步长的处理函数）
# 这里简单使用 torch.round 进行处理，你可以根据实际需求修改 round_sigma 函数
def round_sigma(tensor):
    return torch.round(tensor * 1000) / 1000  # 保留3位小数

t_steps_rounded = round_sigma(t_steps)

# 在 t_steps 的前面加上一个零值
t_steps_final = torch.cat([torch.zeros_like(t_steps_rounded[:1]), t_steps_rounded])

# 绘制图形
plt.plot(t_steps_final.numpy(), marker='o', linestyle='-', color='b')
plt.title("t_steps Over Diffusion Process")
plt.xlabel("Step Index")
plt.ylabel("t_steps Value")
plt.grid(True)
plt.show()
