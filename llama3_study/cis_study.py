from typing import Tuple
import torch
from common import *

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    
    # 首先torch.arange创建了一个tensor，[ 0 , 2 , 4 , . . . , 60 , 62 ]
    # 然后统一除以64，把它变成分数，然后整体作为基础角度的指数，它的shape是(32)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    print("freqs is:", freqs, "shape is: ", freqs.shape)

    # t比较容易理解，也就是绝对位置信息，它的shape是(1024)
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    print("t is: ", t, "shape is: ", t.shape)

    # torch.outer是把一个向量的转置乘以另一个向量：torch.outer(a, b) = a^T * b
    # 于是根据torch.outer运算，我们得到了一个shape为(1024, 32)的tensor。其意义也就是将每一个绝对位置，分配到对应的角度，相乘
    # 直观理解一下，就是每一个绝对位置上，都有32个角度
    # 为什么是这样的呢，回顾计算的公式，对于旋转矩阵，每两个元素为一组，它们乘以的角度是同一个θ，所以这个(1024, 32)
    # 在后续的过程中，就可以reshape成(512, 64)，并且在64的那个维度上，每两个是相同的
    freqs = torch.outer(t, freqs)
    print("freqs is:", freqs, "shape is: ", freqs.shape)

    # torch.polar(abs, angle)利用一个绝对数值和一个角度值，从而在极坐标下构造一个复数张量
    # 即abs∗cos(angle)+abs∗sin(angle)j
    # torch.polar(torch.tensor([1], dtype=torch.float64), torch.tensor([np.pi / 2], dtype=torch.float64))
    # # tensor([6.1232e-17+1.j], dtype=torch.complex128)
    # freqs_cis其实就是需要计算出来的mθ，也就是跟绝对位置相关的旋转的角度，在极坐标下对应的复数tensor
    # 这一步就是在生成我们需要的位置信息
    # 直观理解一下，像是在复平面内，以原点为中心，转了1024组，每一组64个的单位向量，它的shape是(1024, 64)

    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

freqs_cis_result = precompute_freqs_cis(64, 1024)
print("freqs_cis_result is:", freqs_cis_result, "shape is: ", freqs_cis_result.shape)

######################################################################
######################################################################

print("#" * 100)

#   第二个函数reshape_for_broadcast，是把freqs_cis变成和输入的tensor相同的形状
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    # 获取查询张量 xq 的维度数。
    ndim = x.ndim
    print("ndim is: ", ndim)
    print("x shape is: ", x.shape)
    # 确保 xq 至少是二维张量
    assert 0 <= 1 < ndim
    # 这个断言检查确保 freqs_cis 的形状与 xq 的第二维和最后一维匹配。如果不匹配，将打印出错误信息。
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    # 这个方法的作用是为了把freqs_cis变成和输入的tensor相同的形状
    # 需要注意的是，这里的freqs_cis并不是precompute_freqs_cis生成的形状为(1024, 64)的那个tensor
    # 而是根据输入的绝对位置，在(1024, 64)的tensor中，截取了长度为当前seq_len的一部分
    # 代码在Transformer类的forward方法中freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
    # 也就是说，假如当前输入的序列长度是512，那么截取出来的这个新的freqs_cis，形状就是(512, 64)
    # reshape之后，形状就变成了(1, 512, 1, 32)，也就是在每一个位置上，都对应有32个角度
    # 根据上面torch.polar的介绍，当我们固定绝对值(也就是向量的模长)时，角度就可以在笛卡尔坐标系下唯一确定一个复数
    # 这样一来也就是32个复数，即64个特征维度，所以就可以对应的将它融合到每个attention head的64个特征中去了
    # 这行代码创建了一个新的形状列表，其中第二维和最后一维的大小保持不变（d），而其他维度的大小为1。这是为了与 xq 的形状对齐。
    # 代码中的 freqs_cis 必须已经是一个复数张量，且其形状应该与 xq 的第二维和最后一维相匹配。
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    print("shape is: ", shape)
    for i, d in enumerate(x.shape):
        print(i,d)
    return freqs_cis.view(*shape)

# input_tensor = torch.ones(10, 512, 12, 64)
# print(input_tensor)

# result = reshape_for_broadcast(freqs_cis_result, input_tensor)
# print("result is: ", result, " result shape is: ",result.shape)


# 假设 batch_size为2 seq_len固定为512 attention_head的数量为12 每个attention_head的维度为64，那么，对于输入到multi-head attn中的输入x_q的尺寸就是 (2, 512, 12, 64)
# 而freqs_cis其实就是需要计算出来的m\theta也就是跟绝对位置相关的旋转的角度，在极坐标下对应的复数tensor
# 而precompute_freqs_cis就是提前将这些旋转角度对应的tensor给创建出来，并可以重复利用。
# 因为确定了序列的最大长度，所以这个tensor是固定死的。
# 根据后续的数据流我们可以发现，在调用该函数时，传入的两个参数分别是attention_head的维度，以及最大长度的两倍，
# 具象地，也就是64和1024
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # torch.view_as_complex是把一个tensor转为复数形式
    # 比如torch.view_as_complex(torch.Tensor([[1, 2], [3, 4], [5, 6]]))
    # tensor([1.+2.j, 3.+4.j, 5.+6.j])
    
    # 假设输入x_q的尺寸就是(2, 512, 12, 64)
    # 那么这一句操作的reshape，就是把它变成(2, 512, 12, -1, 2)，也就是(2, 512, 12, 32, 2)。x_k同理，略
    # 紧接着把它变成复数形式，也就是变成了(2, 512, 12, 32)的形状。
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    # 把freqs_cis变成和输入的tensor xq_相同的形状
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    # torch.view_as_real是把复数tensor变回实数
    # torch.view_as_real(torch.view_as_complex(torch.Tensor([[1, 2], [3, 4], [5, 6]])))
    # tensor([[1., 2.],
    #         [3., 4.],
    #         [5., 6.]])
    # reshape之后，就是将位置信息融入query和key中
    # 这一步将二者相乘得到的复数tensor，重新转换为实数形式，得到的shape为(2, 512, 12, 32, 2)
    # 然后再flatten成(2, 512, 12, 64)，这样一来，就变回了和最开始x_q相同的形状，也就完成了将位置信息融入到x_q的这一操作，x_k同理
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

xq = torch.randn(2, 1024, 12, 64)
xk = torch.randn(2, 1024, 12, 64)

xq_output, xk_output = apply_rotary_emb(xq, xk, freqs_cis_result)

print("xq is: ", xq, "shape is: ", xq.shape)
print("xk is: ", xk, "shape is: ", xk.shape)

