import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import os
import yaml
from torchvision import transforms
from torchvision.transforms import AutoAugmentPolicy
from tqdm import tqdm
from Models.Resnet import ResNet101
from Models.Mobilenetv2 import MobileNetV2
from Models.Vgg import VGG
from Weapon.WarmUpLR import WarmUpLR
from torch.utils.tensorboard import SummaryWriter

# Read the configuration from the config.yaml file
with open("config.yaml","r") as f:
    config = yaml.load(f,yaml.FullLoader)["Training_seting"]


# Read the configuration from the config.yaml file
batch_size = config["batch_size"]
training_epoch = config["training_epoch"]
num_workers = config["num_workers"]
lr_rate = config["learning_rate"]
warmup_epoch = config["warmup_epoch"]
best_acc = 0


# Set the dataset mean, standard deviation, and input size based on the chosen dataset
if config["dataset"] == "Cifar10":
    dataset_mean = [0.4914, 0.4822, 0.4465]
    dataset_std = [0.2470, 0.2435, 0.2616]
    input_size = 32
elif config["dataset"] == "Cifar100":
    dataset_mean = [0.5071, 0.4867, 0.4408]
    dataset_std = [0.2675, 0.2565, 0.2761]
    input_size = 32

# Check if GPU is available, otherwise use CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Set the paths for dataset, weights, models, and log data
dataset_path = config["dataset_path"]
weight_path = os.path.join(config["weight_path"],config["dataset"],config["models"]) 
filename = config["models"]
log_path = os.path.join(config["experiment_data_path"],config["dataset"],config["models"]) 
writer = SummaryWriter(log_dir=log_path)


# Set the paths for dataset, weights, models, and log data
train_transform = transforms.Compose(
    [
    transforms.RandomCrop(input_size,padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.autoaugment.TrivialAugmentWide(),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
    transforms.Normalize(mean=dataset_mean,std=dataset_std)
    ])
test_transform = transforms.Compose([
    transforms.RandomCrop(input_size,padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=dataset_mean,std=dataset_std)
])


# Load the dataset based on the chosen dataset (Cifar10, Cifar100, or Imagenet) and apply the defined transformations
print("==> Preparing data")
if config["dataset"] == "Cifar10":
    train_set = torchvision.datasets.CIFAR10(dataset_path,train=True,transform=train_transform,download=True)
    test_set = torchvision.datasets.CIFAR10(dataset_path,train=False,transform=test_transform,download=True)
elif config["dataset"] == "Cifar100":
    train_set = torchvision.datasets.CIFAR100(dataset_path,train=True,transform=train_transform,download=True)
    test_set = torchvision.datasets.CIFAR100(dataset_path,train=False,transform=test_transform,download=True)

# Get the number of classes in the dataset
classes = len(train_set.classes)


# Create data loaders for training and testing
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                          shuffle=True, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                         shuffle=False, num_workers=num_workers)

# Create an instance of the selected model (ResNet101, MobileNetV2, or VGG) and transfer it to the chosen device
print("==> Preparing models")
print(f"==> Using {device} mode")
if config["models"] == "ResNet101":
    net = ResNet101(num_classes=classes)
elif config["models"] == "Mobilenetv2":
    net = MobileNetV2(num_classes=classes)
elif config["models"] == "VGG16":
    net = VGG(num_class=classes)
net.to(device)



# Define the loss function and optimizer for training the model
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.SGD(net.parameters(), lr=lr_rate,momentum=config["momentum"],weight_decay=config["weight_decay"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_epoch)
warmup_scheduler = WarmUpLR(optimizer, len(train_loader) * warmup_epoch)


# Validation function
def validation(network,dataloader,file_name,save=True):
    # Iterate over the data loader
    global best_acc
    accuracy = 0
    running_loss = 0.0
    total = 0
    correct = 0
    network.eval()
    with tqdm(total=len(dataloader)) as pbar:
        with torch.no_grad():
            for i, data in enumerate(dataloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)


                # Perform forward pass and calculate loss and accuracy
                outputs = network(inputs)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)
                running_loss += loss.item()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                accuracy = 100 * correct / total
                pbar.update()
                pbar.set_description_str("Acc: {:.3f} {}/{} | Loss: {:.3f}".format(accuracy,correct,total,running_loss/(i+1)))
            
            # Save the model's checkpoint if accuracy improved
            if not os.path.isdir(weight_path):
                os.makedirs(weight_path)
            check_point_path = os.path.join(weight_path,"Checkpoint.pt")
            torch.save({"state_dict":network.state_dict(),"optimizer":optimizer.state_dict()},check_point_path)    
            if accuracy > best_acc:
                best_acc = accuracy
                if save:
                    PATH = os.path.join(weight_path,f"Model@{config['models']}_ACC@{best_acc}.pt")
                    torch.save({"state_dict":network.state_dict()}, PATH)
                    print("Save: Acc "+str(best_acc))
                else:
                    print("Best: Acc "+str(best_acc))
    return running_loss/len(dataloader),accuracy


# Training function
def train(epoch,network,optimizer,dataloader):
    # loop over the dataset multiple times
    running_loss = 0.0
    total = 0
    correct = 0
    network.train()
    with tqdm(total=len(dataloader)) as pbar:
        # Iterate over the data loader
        for i, data in enumerate(dataloader, 0):
            
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if epoch <= warmup_epoch:
                warmup_scheduler.step()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            running_loss += loss.item()            
            
            
            accuracy = 100 * correct / total
            pbar.update()
            pbar.set_description_str("Epoch: {} | Acc: {:.3f} {}/{} | Loss: {:.3f}".format(epoch,accuracy,correct,total,running_loss/(i+1)))


# Training and Testing Loop
print("==> Start training/testing")
for epoch in range(training_epoch + warmup_epoch):
    train(epoch, network=net, optimizer=optimizer,dataloader=train_loader)
    loss,accuracy = validation(network=net,file_name=filename,dataloader=test_loader)
    if (epoch > warmup_epoch):
        scheduler.step()
    writer.add_scalar('Test/Loss', loss, epoch)
    writer.add_scalar('Test/ACC', accuracy, epoch)
writer.close()
print("==> Finish")