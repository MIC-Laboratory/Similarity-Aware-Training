'''MobileNetV2 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math
import numpy as np
import logging

from scipy.spatial.distance import cdist
class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out

    
class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)
        self.mse_loss_list = []
        self.centers = []
        self.ks = []
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.calculate_k()


    def get_k_centers_assign(self):
        for layer in range(len(self.layers)):
            weight = self.layers[layer].conv2.weight.data.clone().cpu().detach().numpy().reshape(self.layers[layer].conv2.weight.shape[0],-1)
            centers_np = self.get_random_centers(weight,self.ks[layer])
            centers_tensor = torch.tensor(centers_np, device=self.device)
            self.centers.append(centers_tensor)
        self.assign_weight_center()

    def calculate_k(self):
        for layer in self.layers:
            output_channel = layer.conv2.weight.shape[0]
            self.ks.append(int(math.sqrt(output_channel//2)))

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def get_centers(self):
        return self.centers

    def assign_weight_center(self):
        self.assigned_center = []
        for layer in range(len(self.layers)):
            data = self.layers[layer].conv2.weight
            centers = self.centers[layer]
            self.assigned_center.append([])
            for i in range(data.shape[0]):
                random_centers_index = np.random.choice(range(len(centers)))
                self.assigned_center[layer].append(random_centers_index)
    def get_mse_loss(self):
        self.mse_loss_list = []
        
        for j,layer in enumerate(self.layers,0):
            data = layer.conv2.weight.reshape(layer.conv2.weight.shape[0],-1)
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

# Setup logger
logging.basicConfig(filename="std.log", 
					format='%(asctime)s %(message)s', 
					filemode='w')
logger=logging.getLogger()
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)   
model = MobileNetV2(num_classes=100)
logger.debug(model.ks)