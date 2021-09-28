import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from load_data import *
from net import *

if __name__ == '__main__':
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    correct = 0
    total = 0
    # train
    for epoch in range(4):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            images, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = net(images)
            # print(outputs)
            # break
            # indice de la valeur max (0 pas face, 1, c'est face)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        # test
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            # indice de la valeur max (0 pas face, 1, c'est face)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
