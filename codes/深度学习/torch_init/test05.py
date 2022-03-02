import torch
print("torch.take")
a = torch.linspace(1, 16, 16).reshape(4, 4)
print(a)

b = torch.take(a, index=torch.tensor([0, 15, 13, 10]))

print(b)
