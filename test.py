import numpy as np
import torch

# xs = []
# for _ in range(10):
#     x = torch.randn(5, 3)
#     xs.append(x.detach().unsqueeze(0))
# xs = torch.cat(xs, dim=0)
# print(xs.size())

# xs = []
# for _ in range(10):
#     x = torch.randn(5, 3)
#     xs.append(x.detach())
# xs = np.array(xs)
# print(xs.shape)

x = torch.rand(2, 3)
print(x.unsqueeze(1).size())
