import  torch

a = torch.rand(4,4)
b = torch.rand(4,4)

print(a)
print(b)


out = torch.where(a>0.5,a,b)
# // 返回a,b中满足条件的值
print(out)
out = torch.where(a>0.5)
# // 返回a,b中满足条件的索引

print(out)


