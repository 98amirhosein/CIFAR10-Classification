import torch
import torchvision
import torch.utils.data as datautils
import matplotlib.pyplot as plt
import torch.nn as to_nn
import torch.nn.functional as to_f
import torch.optim as optim


def matplotlib_imshow(img_tensor, title=None):

    device = torch.device("cpu")
    img_tensor_cpu = img_tensor.to(device)

    std = 0.5
    mean = 0.5

    img_tensor = std * img_tensor_cpu + mean
    img = img_tensor.numpy()
    img = img.transpose(1, 2, 0)
    plt.figure()
    plt.imshow(img)
    if title is not None:
        plt.title(title)


image_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
cifar10_training_set = torchvision.datasets.CIFAR10(root='./dataset_CIFAR10', train=True, transform=image_transforms,
                                                    download=True)
cifar10_test_set = torchvision.datasets.CIFAR10(root='./dataset_CIFAR10', train=False, transform=image_transforms,
                                                    download=True)
trainingset_loader = datautils.DataLoader(dataset=cifar10_training_set, batch_size=4, shuffle=True, num_workers=2)
testset_loader = datautils.DataLoader(dataset=cifar10_test_set, batch_size=4, shuffle=True, num_workers=2)

data_iter = iter(trainingset_loader)
for i in range(3):
    images, labels = data_iter.next()
    matplotlib_imshow(img_tensor=torchvision.utils.make_grid(images), title=['gt: '+classes[i.item()] for i in labels])






class ConvNet(to_nn.Module):

    def __init__(self):

        super(ConvNet, self).__init__()

        self.conv1 = to_nn.Conv2d(in_channels=3, out_channels=9, kernel_size=(5, 5))
        self.conv2 = to_nn.Conv2d(in_channels=9, out_channels=18, kernel_size=(3, 3))
        self.fc1 = to_nn.Linear(in_features=18 * 6 * 6, out_features=100)
        self.fc2 = to_nn.Linear(in_features=100, out_features=40)
        self.fc3 = to_nn.Linear(in_features=40, out_features=10)

    def forward(self, input):
        out = self.conv1(input)
        out = to_f.max_pool2d(to_f.relu(out), (2, 2))
        out = self.conv2(out)
        out = to_f.max_pool2d(to_f.relu(out), (2, 2))
        out = out.view(-1, 18 * 6 * 6)
        out = to_f.relu(self.fc1(out))
        out = to_f.relu(self.fc2(out))
        out = self.fc3(out)
        return out


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device: {}'.format(device))

net = ConvNet()
net.to(device)

criterion = to_nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(10):

    running_loss = 0.0

    for i, data in enumerate(trainingset_loader, 0):

        images, labels = data

        optimizer.zero_grad()

        images = images.to(device)
        output = net(images)

        labels = labels.to(device)
        loss = criterion(output, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if i % 1000 == 999:
            print('epoch {}, mini-batch {}, loss {}'.format(epoch, i + 1, running_loss/1000.0))
            running_loss = 0.0

print('Training Phase is done.')

data_iter = iter(testset_loader)
for i in range(3):
    images, labels_ground_truth = data_iter.next()
    images = images.to(device)
    labels_ground_truth = labels_ground_truth.to(device)
    outputs = net(images)
    _, predicted = torch.max(outputs, dim=1)
    matplotlib_imshow(img_tensor=torchvision.utils.make_grid(images), title=['predicted: '+classes[i.item()] for i in predicted])

correct = 0
total = 0
with torch.no_grad():
    for data in testset_loader:
        images, ground_truth_labels = data
        images = images.to(device)
        ground_truth_labels = ground_truth_labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, dim=1)
        total += ground_truth_labels.size(0)
        correct += (ground_truth_labels==predicted).sum().item()
print('Accuracy on test set is: {}%'.format(correct/total * 100))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testset_loader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of {}: {}%'.format(classes[i], 100 * class_correct[i] / class_total[i]))

plt.show()