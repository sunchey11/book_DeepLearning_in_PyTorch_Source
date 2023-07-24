# 训练模型的完整代码
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root="D:\\pytorch_data\\fashion-mnist\\data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="D:\\pytorch_data\\fashion-mnist\\data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
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

model = NeuralNetwork()
print(model)

learning_rate = 1e-3
batch_size = 64
epochs = 5

# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
def train_loop(dataloader, model, loss_fn, optimizer):
    # 这个得到的是dataloader的总大小，为60000
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        # X有64条数据，X.shape=[64,1,28,28]
        # pred是预测结果,pred.shape=[64,10]
        # 表示64条数据的预测结果，共10个分类，每个分类一个数字（不是概率），
        pred = model(X)

        # loss是一个Tensor，里面是个数字，不是数组
        # loss.item()可以得到这个数字
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            # current表示当前是第几条
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    # 测试数据一共10000条
    size = len(dataloader.dataset)
    # 一共有157个批次，10000/64=156.25
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        
        for X, y in dataloader:
            # X是图片数据，y是实际数据,是图片所属的分类，是一个数字数组
            # X.shape=[64,1,28,28]
            # y.shape=[64]
            # pred是预测结果,pred.shape=[64,10]
            # 表示64条数据的预测结果，共10个分类，每个分类一个数字（不是概率），
            pred = model(X)
            # loss_fn函数结果为一个数字。
            # 为什么要把每次的损失值加起来，因为要求平均损失
            test_loss += loss_fn(pred, y).item()
            # pred.argmax(1) 获得最大值的序号
            # myv.shape=[64],里面的元素是Boolean，表示预测是否正确
            myv = (pred.argmax(1) == y)
            # True转为1，False转为0
            # 所以myv2里的元素为0和1
            myv2 = myv.type(torch.float)
            correct += myv2.sum().item()
    # 总共的损失，除以总次数，得到平均损失
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
# 第一种存法，只存参数，生成一个两兆的文件
torch.save(model.state_dict(), 'mymodel_state_dict.pth')
# 第二种存法，整个都存( structure of this class和参数)
# 生成一个两兆的文件
torch.save(model, 'mymodel_all.pth')
print("Done!")