import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
model = NeuralNetwork().to(device)
print(model)
X = torch.rand(1, 28, 28, device=device)
logits = model(X)
# logits.shape = [1, 10]，这是linear_relu_stack的处理结果
print(logits)
print(logits.shape)

# 将数据转为0到1之间的概率，总和为1
pred_probab = nn.Softmax(dim=1)(logits)
print(pred_probab)
print(type(pred_probab))

# https://pytorch.org/docs/stable/generated/torch.argmax.html#torch.argmax
# 返回最大值的索引
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")