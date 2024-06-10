# layernorm是大模型也是transformer结构中最常用的归一化操作，简而言之，它的作用是对特征张量按照某一维度或某几个维度进行0均值，1方差的归一化操作。
# 根据这个公式，需要理解两点：
# 1. 均值和方差怎么计算
# 一般输入都是batch_size,seq_length,embedding。均值和方差是分别计算每个token的均值和方差
# 2. gamma和beta是layernorm的两个可训练参数，具体维度是多少
# gamma和beta的维度是和embedding的维度保持一致，相当于对向量的每个维度进行缩放和平移

import torch
import torch.nn as nn

# 设备
DEVICE='cuda' if torch.cuda.is_available() else 'cpu' 
 
hidden_size = 3
layer_norm_eps = 1e-5
#带参数
layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps).to(DEVICE, dtype=torch.float)
#不带参数
layernorm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps, elementwise_affine=False).to(DEVICE)
 
#shape=(2, 2, 3)
hidden_states = torch.tensor([[[1, 2, 3],[2, 3, 1]],[[3, 1, 2],[4, 2, 5]]]).to(DEVICE, dtype=torch.float)
 
hidden_states = layernorm(hidden_states)

print(hidden_states)

########################################################
########################################################

print("=" * 50)
print("RMSNorm metnod in Llama3")

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    
hidden_size = 4
layer_norm_eps = 1e-5

hidden_states = torch.tensor([[[1, 2, 3, 4],[2, 3, 1, 2]],[[3, 1, 2, 1],[4, 2, 5, 3]]]).to(DEVICE, dtype=torch.float)

rmsnorm = RMSNorm(hidden_size, eps=layer_norm_eps).to(DEVICE, dtype=torch.float)

hidden_states = rmsnorm(hidden_states)

print(hidden_states)

print("=" * 50)

########################################################
########################################################

hidden_size = 3
layer_norm_eps = 1e-5

hidden_states = torch.tensor([[1, 2, 3],[2, 3, 1],[3, 1, 2],[4, 2, 3]]).to(DEVICE, dtype=torch.float)

rmsnorm = RMSNorm(hidden_size, eps=layer_norm_eps).to(DEVICE, dtype=torch.float)

hidden_states = rmsnorm(hidden_states)

print(hidden_states)

print("=" * 50)

