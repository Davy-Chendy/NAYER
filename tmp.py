from torchvision import datasets

# 自动下载并解压到 ./data 目录
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True)