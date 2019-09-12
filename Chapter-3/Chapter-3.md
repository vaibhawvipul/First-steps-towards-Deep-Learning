# Chapter 3
## How to make a computer see?

### Introduction

Computer Vision has been one of the most captivating areas of interest for researchers. One can say that a lot of progress which has been made in the field of deep learning was motivated by the challenge to solve some computer vision problems like object detection, classification, etc. 

Honestly, we still don’t understand how we see! Our eyes in sync with our brain does amazingly complicated tasks(like completely ignoring our own nose while seeing) really well. Imagine yourself wearing a helmet while driving a motorbike, during the rainy season when droplets start to slide down on the visor our vision system is completely capable of ignoring that and gives us proper contextual information which helps us in driving. 

The “Intelligent” vision systems which we design works really well with a lot of constraints. Anyways, the field is progressing rapidly and we are not only stuck in doing a single job really well like detection or classification but we are trying to get contextual information with which the machine can reason!

---

### Convolutional Neural Networks

As promised, this book will help the reader to take their first steps in deep learning. So in this section we will be overlooking a lot of mathematics while focusing on intuition.

In deep learning world, Convolutional Neural Networks (or ConvNets or CNNs) have become a standard way to solve any problem related to images. The best part about CNNs are that they require much less preprocessing as compared to their predecessors. There are almost no hard coded features required.

Let us take a moment here and understand how a computer sees an image. To a computer an image is nothing but a multidimensional array(3-d if image is RGB). The values range from 0-255. 
To a computer, image is nothing but a collection of pixel values.

#### Can’t fully connected networks process images?

It is a common argument that why can’t we simply stretch the multidimensional array and flatten it out and pass it through a fully connected layer.

*A pic would be nice here, working on it*

Again not getting into mathematics but if we try to understand intuitively, if we reduce the dimension of any multidimensional array to a single dimension we will surely lose some information, right? 

Note - There are some dimensionality reduction techniques which might not lose data always. 

The following is the example image = 
![alt text](https://github.com/vaibhawvipul/First-steps-towards-Deep-Learning/blob/master/Chapter-3/assests/img_3d.jpg "Figure")
source - ["Computer History Museum" by Scott Beale is licensed under CC BY-NC 2.0](https://ccsearch.creativecommons.org/photos/52b18712-554e-4aff-8c87-d74187d92a07) 

Let us try to visualize a three dimensional array.

```
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

image = "Chapter-3/assests/img_3d.jpg"

img=mpimg.imread(image)
print(img.shape) #shape will be (333, 499, 3) where 3 represents 3-dimension

imgplot = plt.imshow(img)
plt.show()

```

Following will be the output of the code - 

![alt text](https://github.com/vaibhawvipul/First-steps-towards-Deep-Learning/blob/master/Chapter-3/assests/matplotlib-3d-visual.png "Output")

If we try to flatten the image we can only visualize it via a histogram.

```
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

image = "Chapter-3/assests/img_3d.jpg"

img=mpimg.imread(image)
print(img.shape)

img = img.flatten()
print(img.shape) # shape of the image will be (498501,) now.
 
plt.hist(img)
plt.show()
```

Following is the output - 

![alt text](https://github.com/vaibhawvipul/First-steps-towards-Deep-Learning/blob/master/Chapter-3/assests/img-flat-hist.png "Output Hist")

we clearly cannot reason much by seeing th output of histogram. 

In cases of extremely basic binary images like MNIST dataset, the multi-layer perceptron or a fully connected network can give decent results but in real world scenario we need a network which can take 3-d images as input and eventually extract relevant features out of it.

A ConvNet is able to successfully capture the Spatial features/context in an image by applying multiple filters or kernels.

A convolutional layer is built by using basically three components - 
- convolutional layer
- pooling layer
- fully connected layers

Usually a convNet takes an input image and gives score/class probabilities as an output.

Note - understanding convolutional layer will require some basic mathematics which we will ignore for now. Feel free to raise PRs related to this.

--- 

### Writing a Convolutional Neural Network in pyTorch

```
#importing necessary libraries 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Note - There is a difference between “nn.Conv2d” and “nn.functional.conv2d”. The “nn.Conv2d” is meant 
# to be used as a convolutional layer directly. However “nn.functional.conv2d” is meant to be used when 
# you want your custom convolutional layer logic.

# we will use torchvision library to download and add transformations to our data
import torchvision as tv 
import torchvision.transforms as transforms 

# our transformation pipeline
transform = transforms.Compose([tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]) 

trainset = tv.datasets.CIFAR10(root="./data",train=True,download=True,transform=transform) 
dataloader = torch.utils.data.DataLoader(trainset,batch_size=4, shuffle=False, num_workers=4)

# Defining our model 
class OurModel(nn.Module):
    def__init__(self):
        super(OurModel,self).__init__()
        self.conv1 = nn.Conv2d(3,6,5) 
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = OurModel()
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=0.001,weight_decay= 1e-6, momentum = 0.9, nesterov = True)

# training
for epoch in range(2):
    running_loss= 0.0

    for i,data inenumerate(dataloader,0):
        inputs, labels = data
        optimizer.zero_grad()

        # forward prop
        outputs = net(inputs)
        loss = loss_func(outputs, labels)

        # backprop
        loss.backward() # compute gradients
        optimizer.step() # update parameters

        # print statistics
        running_loss += loss.item()
        if i %2000==1999: # print every 2000 mini-batches
            print('[epoch: %d, minibatch: %5d] loss: %.3f'%(epoch +1, i +1, running_loss /2000))
            running_loss = 0.0

print("Training finished!")
```

That's is how you can write and quickly run a basic convolutional network in pytorch.

Now let's verify our model

```
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

dataiter = iter(testloader) 
images, labels = dataiter.next()

outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4))) 
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
```