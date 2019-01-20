import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torchvision


def imshow(img):
    for i in range(int(img.size(1) / 36)):
        t = img[:, 36 * i:36 * (i + 1), :]
        t -= t.min()
        t /= (t.max() - t.min())
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 16 * 5 * 5)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def visual_conv1(self, x, map_id):
        x = self.conv1(x)
        x = F.relu(x)

        pool_kernel_size = self.pool.kernel_size
        switches = []
        for image_id in range(x.size(0)):
            for tx in range(int(x.size(2)/pool_kernel_size)):
                for ty in range(int(x.size(3)/pool_kernel_size)):
                    posX = pool_kernel_size * tx
                    posY = pool_kernel_size * ty
                    vx, sx = torch.max(
                        x[image_id, map_id, posX:posX + pool_kernel_size, posY:posY + pool_kernel_size], 0)
                    vy, sy = torch.max(vx, 0)
                    sx = sx[sy]
                    switches.append((image_id, posX+sx.item(),
                                     posY+sy.item(), vy.item()))

        # pooling...

        # unpooling
        x[:, map_id].zero_()

        for image_id, sx, sy, v in switches:
            x[image_id, map_id, sx, sy] = v

        # relu
        x = F.relu(x)

        # deconv
        deconv1 = nn.ConvTranspose2d(
            in_channels=1, out_channels=3, kernel_size=5)
        deconv1.weight = nn.Parameter(self.conv1.weight[map_id:map_id+1])
        x = x[:, map_id:map_id + 1]
        x = deconv1(x)
        return x


if __name__ == "__main__":
    net = LeNet()
    net.load_state_dict(torch.load('lenet_cifar10'))

    trainset = torchvision.datasets.CIFAR10(root='../Datasets/', train=True,
                                            download=True, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=8,
                                              shuffle=True, num_workers=2)
    dataiter = iter(trainloader)
    images, _ = dataiter.next()
    xs = [torchvision.utils.make_grid(images)]
    with torch.no_grad():
        for i in range(6):
            xs.append(torchvision.utils.make_grid(net.visual_conv1(images, i)))
        xs = torch.cat(xs, 1)
        imshow(xs)
