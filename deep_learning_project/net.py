import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # 32 * 32
        self.pool = nn.MaxPool2d(2, 2)  # 16 * 16 (conv1) puis 6 * 6 (conv2)
        self.conv2 = nn.Conv2d(6, 16, 5)  # 12 * 12
        self.fc1 = nn.Linear(16 * 6 * 6, 32)  # en sortie 32
        self.fc2 = nn.Linear(32, 16)  # sortie 16
        self.fc3 = nn.Linear(16, 2)  # sortie 2

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # reshape the tensor: -1 flatten the tensor (16 channel et 6 * 6 en dimension)
        # on se retrouve avec une seule ligne avec une valeur entre 0 et 1
        x = x.view(-1, 16 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
