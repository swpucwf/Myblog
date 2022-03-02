import torch
print("torch.stack")
a = torch.linspace(1, 6, 6).reshape(2, 3)
b = torch.linspace(7, 12, 6).reshape(2, 3)
print(a, b)
out = torch.stack((a, b), dim=2)
print(out)
print(out.shape)

print(out[:, :, 0])
print(out[:, :, 1])