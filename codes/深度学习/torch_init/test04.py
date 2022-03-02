import torch
print("torch.masked_index")
a = torch.linspace(1, 16, 16).reshape(4, 4)
mask = torch.gt(a, 8)
print(a)
print(mask)
out = torch.masked_select(a, mask)
print(out)