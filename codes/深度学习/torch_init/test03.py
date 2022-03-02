import torch

print("torch.gather")
a = torch.linspace(1, 16, 16).reshape(4, 4)

print(a)



out = torch.gather(a, dim=0,
             index=torch.tensor([[0, 1, 1, 1],
                                 [0, 1, 2, 2],
                                 [0, 1, 3, 3]]))
print(out)
print(out.shape)
#注：从0开始，第0列的第0个，第一列的第1个，第二列的第1个，第三列的第1个，，，以此类推
#dim=0, out[i, j, k] = input[index[i, j, k], j, k]
#dim=1, out[i, j, k] = input[i, index[i, j, k], k]
#dim=2, out[i, j, k] = input[i, j, index[i, j, k]]