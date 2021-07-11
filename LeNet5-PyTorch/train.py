from model import Model
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
import torchvision

Batch_size = 100
Download = False

if __name__ == '__main__':
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./', train=True, download=Download,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])), batch_size=Batch_size, shuffle=True)
    # Normalize()转换使用的值0.1307和0.3081是MNIST数据集的全局平均值和标准偏差，这里将它们作为给定值
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./', train=False, download=Download,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))])), batch_size=Batch_size, shuffle=True)

    model = Model()
    sgd = SGD(model.parameters(), lr=1e-1)
    cost = CrossEntropyLoss()

    for _epoch in range(10):
        model.train()
        for idx, (train_x, train_label) in enumerate(train_loader):
            sgd.zero_grad()
            predict_y = model(train_x.float())
            loss = cost(predict_y, train_label.long())
            if idx % 100 == 0:
                print('epoch: {}  idx:{} loss: {}'.format(_epoch+1 ,idx, loss.sum().item()))
            loss.backward()
            sgd.step()

        correct = 0
        sum = 0
        model.eval()
        for idx, (test_x, test_label) in enumerate(test_loader):
            predict_y = model(test_x.float()).detach()
            predict_ys = np.argmax(predict_y, axis=-1)
            label_np = test_label.numpy()
            predit = predict_ys == test_label
            correct += np.sum(predit.numpy(), axis=-1)
            sum += predit.shape[0]

        print('accuracy: {:.2f}'.format(correct / ((idx+1)*len(test_loader))))
    torch.save(model, 'mnist_{:.2f}.pkl'.format(correct / sum))
