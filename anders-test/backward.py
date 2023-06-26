# backward看不懂，写个程序试一下
import torch
x = torch.tensor([[2., -1.], [1., 1.]], requires_grad=True)
print(x)
y = x.pow(2)
print(y)
out = y.mean()
print(out)
out.backward()
print(x.grad)