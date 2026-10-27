# 本文件为实验一的CPU版本：在没有GPU的环境（如普通笔记本）中使用本文件；有GPU时使用model.py即可。
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from MyOptimizer import SGDOptimizer, AdamOptimizer

class Net(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)

        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, num_classes)

    def forward(self, x):
        """
        Args:
            x: (b, 1, 28, 28)
        """
        out = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        out = F.max_pool2d(F.relu(self.conv2(out)), (2, 2))
        # flatten the feature map
        out = out.flatten(1)
        # fc layer
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out

def train(model, dataloader, optimizer, loss_fn, num_epochs=1):
    print("Start training ...")
    loss_total = 0.
    model.train()
    # 如需使用PyTorch Profiler分析训练过程中的时间开销（详见指导书对应章节），
    # 安装 tensorboard 和 torch-tb-profiler 后，取消如下注释，并将下方训练循环放入 with 块内，
    # 且在每一步（每个batch）训练结束后调用一次 prof.step()（CPU环境只保留CPU活动即可）：
    # profiler_config = {
    #     'activities': [torch.profiler.ProfilerActivity.CPU],
    #     'schedule': torch.profiler.schedule(wait=1, warmup=1, active=10, repeat=3),
    #     'on_trace_ready': torch.profiler.tensorboard_trace_handler('./log/profiler'),
    #     'record_shapes': True,
    #     'profile_memory': True,
    #     'with_stack': True}
    # with torch.profiler.profile(**profiler_config) as prof:
    #     ...  # 原训练循环，每步结束后调用 prof.step()
    for epoch in range(num_epochs):
        for i, batch_data in enumerate(dataloader):
            inputs, labels = batch_data

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            loss_total += loss.item()
            if i % 20 == 19:
                print('epoch: %d, iters: %5d, loss: %.3f' % (epoch + 1, i + 1, loss_total / 20))
                loss_total = 0.0

    print("Training Finished!")

def test(model: nn.Module, test_loader):
    # test
    model.eval()
    size = len(test_loader.dataset)
    correct = 0
    print("testing ...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            output = model(inputs)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).sum().item()
    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(
        correct, size,
        100 * correct / size))

def main():
    model = Net(in_channels=1, num_classes=10)

    DATA_PATH = "./data"

    transform = torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor()]
                )

    train_set = torchvision.datasets.MNIST(DATA_PATH, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(DATA_PATH, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = SGDOptimizer(model.parameters(), lr=0.01)


    train(model, train_loader, optimizer, loss_fn)
    test(model, test_loader)

if __name__ == "__main__":
    main()
