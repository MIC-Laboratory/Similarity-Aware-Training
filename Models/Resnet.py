'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math
import numpy as np
from scipy.spatial.distance import cdist

# Setup logger
logging.basicConfig(filename="std.log", 
					format='%(asctime)s %(message)s', 
					filemode='w')
logger=logging.getLogger()
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)   

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.mse_loss_list = []
        self.centers = []
        self.ks = []
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        layersx = [
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4
        ]
        self.layers = []
        for layers in layersx:
            for layer in range(len(layers)):
                self.layers.append(layers[layer].conv2)
        self.calculate_k()
        for layer in range(len(self.layers)):
            weight = self.layers[layer].weight.data.clone().cpu().detach().numpy().reshape(self.layers[layer].weight.shape[0],-1)
            centers_np = self.get_random_centers(weight,self.ks[layer])
            centers_tensor = torch.tensor(centers_np, device=self.device)
            self.centers.append(centers_tensor)
        self.assign_weight_center()
    
    def calculate_k(self):
        for layer in self.layers:
            output_channel = layer.weight.shape[0]
            self.ks.append(int(math.sqrt(output_channel//2)))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def get_centers(self):
        return self.centers

    def assign_weight_center(self):
        self.assigned_center = []
        for layer in range(len(self.layers)):
            data = self.layers[layer].weight
            centers = self.centers[layer]
            self.assigned_center.append([])
            for i in range(data.shape[0]):
                random_centers_index = np.random.choice(range(len(centers)))
                self.assigned_center[layer].append(random_centers_index)
    def get_mse_loss(self):
        self.mse_loss_list = []
        for j,layer in enumerate(self.layers,0):
            data = layer.weight.reshape(layer.weight.shape[0],-1)
            centers = self.centers[j][self.assigned_center[j]]
            losses = F.smooth_l1_loss(data, centers, reduction='none').sum(dim=1)
            self.mse_loss_list.append(losses.sum())

        self.mse_loss_list = torch.stack(self.mse_loss_list).to(self.device)
        return self.mse_loss_list
    def get_random_centers(self,data, k):
        # Initialize the centers list with one random point from the data
        centers = [data[np.random.choice(len(data))]]
        for _ in range(k - 1):
            distances = cdist(centers, data, 'euclidean')
            farthest_point = data[np.argmax(np.min(distances, axis=0))]
            centers.append(farthest_point)
        return np.array(centers)

def ResNet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2],num_classes)


def ResNet34(num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3],num_classes)


def ResNet50(num_classes):
    return ResNet(Bottleneck, [3, 4, 6, 3],num_classes)


def ResNet101(num_classes):
    return ResNet(Bottleneck, [3, 4, 23, 3],num_classes)


def ResNet152(num_classes):
    return ResNet(Bottleneck, [3, 8, 36, 3],num_classes)


def test():
    net = ResNet101(num_classes=100)
    net.to(net.device)
    y = net(torch.randn(1, 3, 32, 32).to(net.device))
    logger.debug(net.ks)
    # logger.debug(net.get_mse_loss())
test()