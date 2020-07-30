#Cifar-10 classification

In this project I'm going to classify images from the CIFAR-10 dataset by CNN (Convolutional Neural Network).
The dataset consists of airplanes, dogs, cats, and other objects. 
You'll preprocess the images, then train a convolutional neural network on all the samples. 
The images need to be normalized and the labels need to be one-hot encoded. 
Cifar-10 contain 10 class:
 1. Airplain
 2. Car
 3. Bird
 4. Cat
 5. Deer
 6. Dog
 7. Frog
 8. Horse
 9. Ship
 10. Truck
    
Each class contains 5,000 training data and 1,000 test data, and the entire database contains 60,000 images. 
The data size is about 160 MB. Each image in this dataset is 32 x 32 pixels. 
    
At first we must download dataset .
    
    wget  https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    tar -xvzf cifar-10-python.tar.gz
    
Since the datasheet is an image, we use a convolutional network because such networks have very good results 
on the images and learn the spatial information in the image well.
Since the goal is to achieve a reasonable accuracy and work with convolutional neural networks and the Paytour framework, 
we use a light and simple convolutional network that achieves relatively good accuracy and is easy to train.
Convolution network that we use is as follows:


* A canonization layer that takes a 3-channel input and a 9-channel output with a kernel size of 5 x 5 and uses the non-linear ‌ReLu function.
* A P Max Pooling layer with a kernel size of 2 * 2.
* A canonization layer that takes an input with 9 channels and an output with 18 channels and its kernel size is 3 * 3 and uses the nonlinear function eReLu.
* A P Max Pooling layer whose kernel size is 2 * 2.
* A Fully Connected layer that has 100 neurons and uses the nonlinear ReLu function.
* A Fully Connected layer that has 40 neurons and uses the nonlinear ReLu function.
* A Fully Connected layer that has 10 neurons equal to the number of classes.

First, we import the packages used:
    
    import torch
    import torchvision
    import numpy
    import os
    import torch.utils.data as datautils
    import matplotlib.pyplot as plt
    import torch.nn as to_nn
    import torch.nn.functional as to_f
    import torch.optim as optim`
    
torch and torchvision are two packages for working with neural networks and performing tasks related to computer vision (vision).

Then we read the database:
    
    # transforms that should be done of each image
    # 1- convert from numpy to tensor
    # 2- normalize pixel values to real value range [-1,1], for better and easier training
    image_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
     
     
    # load training and test set
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    cifar10_training_set = torchvision.datasets.CIFAR10(root='./dataset_CIFAR10', train=True, transform=image_transforms,
                                                        download=True)
    cifar10_test_set = torchvision.datasets.CIFAR10(root='./dataset_CIFAR10', train=False, transform=image_transforms,
                                                        download=True)
    trainingset_loader = datautils.DataLoader(dataset=cifar10_training_set, batch_size=4, shuffle=True, num_workers=2)
    testset_loader = datautils.DataLoader(dataset=cifar10_test_set, batch_size=4, shuffle=True, num_workers=2)



In the above code, transforms.ToTensor is a function that converts a normal array of numpy to a Tensor. Peturch uses a tensor and does not use normal and numpy arrays.
 The difference is that the tensor can use GPU. The transforms.Normaalize function transfers the value of pixels from -1 to 1, depending on the input given to it, 
 and this normalization makes network training easier. The compose function combines the above two functions. Pythorch has functions and capabilities for reading some popular datasets, 
 and one of these datasets is the CIFAR10 database that we used above. If the data is not available in the specified path, it will download it automatically.
 DataLoader allows mini-batch access to dataset data and also performs shuffling.
 
     
    def matplotlib_imshow(img_tensor, title=None):
        """
        Get and image in tensor form and display it.
        :param img_tensor:
        :return:
        """
        device = torch.device("cpu")
        img_tensor_cpu = img_tensor.to(device)
     
        std = 0.5
        mean = 0.5
     
        img_tensor = std * img_tensor_cpu + mean
        # convert tensor to numpy
        img = img_tensor.numpy()
        # in pytorch image is (3, H, W)
        # normal form is (H, W, 3)
        img = img.transpose(1, 2, 0)
        plt.figure()
        plt.imshow(img)
        if title is not None:
            plt.title(title)

The above function displays an image that is a Tensor. It is initially specified that the tensor is to be processed with CPU.
 The values -1 to 1 are then transferred to the range 0 to 1, and then the tensor is converted to a numpy array, and then the image is converted from 3HW to HW3.
 The reason for this is the difference between the image in Peturch and other libraries and standards, and at the end the image is drawn.
 
 
    # show some samples of traingset
    data_iter = iter(trainingset_loader)
    for i in range(3):
        images, labels = data_iter.next()
        matplotlib_imshow(img_tensor=torchvision.utils.make_grid(images), title=['gt: '+classes[i.item()] for i in labe
        
Then

    # define a convolutional neural network
    class ConvNet(to_nn.Module):
     
        # constructor of model
        def __init__(self):
            # constructor of parent class
            super(ConvNet, self).__init__()
            # learnable parameter of model (convolution and fully connected layers)
            self.conv1 = to_nn.Conv2d(in_channels=3, out_channels=9, kernel_size=(5, 5))
            self.conv2 = to_nn.Conv2d(in_channels=9, out_channels=18, kernel_size=(3, 3))
            self.fc1 = to_nn.Linear(in_features=18 * 6 * 6, out_features=100)
            self.fc2 = to_nn.Linear(in_features=100, out_features=40)
            self.fc3 = to_nn.Linear(in_features=40, out_features=10)
     
        # feedforward
        def forward(self, input):
            # convolutional layer 1
            out = self.conv1(input)
            # pooling kernel size(2,2) and relu non linearity
            out = to_f.max_pool2d(to_f.relu(out), (2, 2))
            # convolutional layer 2
            out = self.conv2(out)
            # pooling kernel size(2,2) and relu non linearity
            out = to_f.max_pool2d(to_f.relu(out), (2, 2))
            # flatten
            out = out.view(-1, 18 * 6 * 6)
            # fully connected 1 with relu non linearity
            out = to_f.relu(self.fc1(out))
            # fully connected 2 with relu non linearity
            out = to_f.relu(self.fc2(out))
            # fully connected 3
            out = self.fc3(out)
            return out


In the code above, we create a convolutional neural network as described above. The class we write must be the parent class to_nn.Module.
 In the constructor, we first call the parent class constructor, then define the layers of the network that have learnable parameters.
In the forward function, we specify the processes that are given to an input given to the network. 
Usually we write layers that do not have learnable parameters in this section and we avoid writing them in the class constructor.            


    # if gpu available use gpu otherwise use cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device: {}'.format(device))
    
In this part of the code we specify whether we want to use the CPU or we want to use the GPU.
 If GPU is available, the above code indicates that we want to use GPU.

    # create and instance of neural network
    net = ConvNet()
    net.to(device)
    
In the above code, we create an object from the network class we have created and specify it to use GPU or CPU.

    # criterion(loss) and optimization algorithm
    # cross entropy loss
    criterion = to_nn.CrossEntropyLoss()
    # adam optimization algorithm, learning rate 0.001 and momentum 0.9
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
Our neural network needs a loss function, we use the Cross Entropy cost function, which usually gives good results for categorization. 
We also need the optimization algorithm and for this we have chosen the Stochastic Gradient Descent algorithm.
    
    # train the convolutional neural network
    # train for number of epochs
    for epoch in range(10):
     
        # loss of epoch
        running_loss = 0.0
     
        # iterate over all batches
        for i, data in enumerate(trainingset_loader, 0):
     
            images, labels = data
     
            # makes all gradients zero
            optimizer.zero_grad()
     
            # feedforward
            images = images.to(device)
            output = net(images)
     
            # loss
            labels = labels.to(device)
            loss = criterion(output, labels)
     
            # backpropagation
            loss.backward()
     
            # update parameters
            optimizer.step()
     
            running_loss += loss.item()
     
            if i % 1000 == 999:
                print('epoch {}, mini-batch {}, loss {}'.format(epoch, i + 1, running_loss/1000.0))
                running_loss = 0.0
     
    print('Training Phase is done.')
    

In the above code, network training is performed. We teach the network for 10 ipak. zero_grad resets the network gradients to zero If we do not do this, 
the gradients will add up and an error will occur.
 The backward function also does the backpropagation and calculates the gradients.
 
     
     
    # show qualitative result of training
    data_iter = iter(testset_loader)
    for i in range(3):
        images, labels_ground_truth = data_iter.next()
        images = images.to(device)
        labels_ground_truth = labels_ground_truth.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, dim=1)
        matplotlib_imshow(img_tensor=torchvision.utils.make_grid(images), title=['predicted: '+classes[i.item()] for i in predicted])
    
    # accuracy on test set
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
    
    # accuracy of each class in test set
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
    
    # show qualitative result of training
    data_iter = iter(testset_loader)
    for i in range(3):
        images, labels_ground_truth = data_iter.next()
        images = images.to(device)
        labels_ground_truth = labels_ground_truth.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, dim=1)
        matplotlib_imshow(img_tensor=torchvision.utils.make_grid(images), title=['predicted: '+classes[i.item()] for i in predicted])
     
    # accuracy on test set
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
     
    # accuracy of each class in test set
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
    
In the above code, after the network training, the network prediction for several inputs selected from the test suite is displayed and then the accuracy is displayed on the test suite.