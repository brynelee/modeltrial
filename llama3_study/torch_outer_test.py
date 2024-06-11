import torch

a = torch.tensor([1,2,3])
b = torch.tensor([3,1,4,5,6])

c = torch.outer(a,b)

print(c)