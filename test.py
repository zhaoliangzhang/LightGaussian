import torch
from torch import nn

mask_activation = torch.sigmoid
x = torch.randn(3,3)
x = mask_activation(x)
print(x)
# m = nn.Threshold(0.5, 0)
x = nn.Threshold(0.5, 0)(x)
print(x)
print(torch.numel(x))
print(torch.count_nonzero(x).cpu().detach().numpy())
print(1 - torch.count_nonzero(x).cpu().detach().numpy()/torch.numel(x))