import torch
x = torch.tensor([[2., -1.], [1., 1.]], requires_grad=True)
print(x)
y = x.pow(2)
print(y)
out = y.mean()
print(out)
out.backward()
print(x.grad)