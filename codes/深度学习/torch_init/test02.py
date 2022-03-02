import torch
print("torch.index_select")
a = torch.rand(4, 4)
print(a)


out = torch.index_select(a, dim=0,
                   index=torch.tensor([0, 3, 2]))

#dim=0按列，index取的是行
print(out)
print(out.shape)