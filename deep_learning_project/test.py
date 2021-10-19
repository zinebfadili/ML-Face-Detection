import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from load_data import *
from net import *
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchsampler import ImbalancedDatasetSampler
from PIL import Image


def trainNet(train_loader, net):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    correct = 0
    total = 0
    print("size of train_loader " + str(len(train_loader.dataset)))
    # train
    zeros = 0
    ones = 0
    for epoch in range(3):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            images, labels = data
            for label in labels:
                if label == 0:
                    zeros += 1
                else:
                    ones += 1
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = net(images)
            # indice de la valeur max (0 pas face, 1, c'est face)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0

    print("ZEROS " + str(zeros))
    print("ONES " + str(ones))
    return net


def testNet(test_loader, net):
    total = 0
    correct = 0
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


def bootstrapNet(net):
    transform = transforms.Compose(
        [transforms.Grayscale(),
         transforms.ToTensor(),
         transforms.Normalize(mean=(0,), std=(1,))])

    batch_size = 32

    threshold = 0.8
    ceil = 0.2
    train_dir = './train_images_bootstrap'
    test_dir = './test_images_bootstrap'

    while threshold > 0.2:
        print("threshold " + str(threshold))

        # get equal amount of data
        # load traindata
        train_data = torchvision.datasets.ImageFolder(
            train_dir, transform=transform)

        # get indices of train data and shuffle them
        num_train = len(train_data)
        indices_train = list(range(num_train))
        np.random.shuffle(indices_train)

        # create balancer sampler for train data
        train_sampler = ImbalancedDatasetSampler(
            train_data, indices=indices_train)

        # these are our own textures
        test_data = torchvision.datasets.ImageFolder(
            test_dir, transform=transform)

        # get loaders
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, sampler=train_sampler, num_workers=1)

        print("Size of train loader in bootstrap: " +
              str(len(train_loader.dataset)))

        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=1, shuffle=True, num_workers=1)

        # train the net
        net = trainNet(train_loader, net)

        # test for our textures and store the indices of images that
        # the net got wrong > threshold
        false_images = []
        with torch.no_grad():
            idx = 0
            for data in test_loader:
                image, label = data
                output = net(image)
                # indice de la valeur max (0 pas face, 1, c'est face)
                proba = output.data[0][1]
                if proba >= threshold:
                    image_array = image.cpu().numpy()
                    image_array = np.array(image_array*255, dtype='int8')
                    image_to_save = Image.fromarray(
                        image_array.reshape(36, 36))
                    image_to_save.save(train_dir+"/0/img"+str(idx)+".pgm")
                idx += 1

        threshold -= 0.2
    return net


if __name__ == '__main__':
    net = Net()
    net = trainNet(train_loader, net)
    testNet(test_loader, net)
    torch.save(net.state_dict(), './model_without_bootstrap.pth')
    # bootstrap the net
    print("bootstrapping the net")
    net = bootstrapNet(net)
    print("bootstrapping the net")
    testNet(test_loader, net)
    torch.save(net.state_dict(), './model_with_bootstrap.pth')
