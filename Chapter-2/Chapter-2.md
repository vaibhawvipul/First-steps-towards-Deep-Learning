# Chapter 2
## Introduction to Pytorch

The barrier to entry is getting lower each day for the field of Deep Learning. There were times when people used to write stuff from scratch every time when they had to implement any idea. Today, we have got great frameworks like pytorch and keras(higher level api based on tensorflow) where we can easily build models and verify our ideas. 

Note - If you don’t have a programming background then you can skip this chapter. This chapter assumes that the reader is versed with Python.

---

### Why pyTorch?

Pytorch is a python based scientific computing framework which also helps us in designing deep neural networks. Being python friendly it makes it really easy beginners to start coding right away. Python has kind-of become lingua franca for the machine learning world. 

```
“I've been using pytorch a few months now and I've never felt better. I have more energy. my skin is clearer. my eyesight has improved.”

— Andrej Karpathy
```

Some of the key things which I personally like is - 
- Simple and Intuitive APIs
- Computational Graph

Dynamic computational graph of pytorch makes it really intuitive to code. Since the graph is not static as compared to tensorflow, it makes it easier for deep neural network architects to change the behaviour on fly. 

```
“An additional benefit of Pytorch is that it allowed us to give our students a much more in-depth understanding of what was going on in each algorithm that we covered. With a static computation graph library like TensorFlow, once you have declaratively expressed your computation, you send it off to the GPU where it gets handled like a black box. But with a dynamic approach, you can fully dive into every level of the computation, and see exactly what is going on.” 

- Jeremy Howard

```

There are a lot of benchmarks showing that pytorch is faster than keras and sometimes comparable to tensorflow.  

A confession here, I love cpp more than python. PyTorch is deeply integrated with the C++ code, and it shares some C++ backend with torch. One can further speed up things by using C++ because they will be closer to machine. Although one can write C++ code in tensorflow also.

Pytorch’s coding style is imperative rather than declarative(which tensorflow has). It makes things intuitive because a lot of people are used to the imperative coding style. If you are someone who wants to prototype ideas quickly then Pytorch will definitely increase developer productivity.

Debugging is often saviour of a developer. Pytorch brings ease in debugging as compare to other frameworks out there. 

The important thing here to note is PyTorch uses different backends for each computation devices rather than using a single back-end. It uses tensor backend TH for CPU and THC for GPU. While neural network backends such as THNN and THCUNN for CPU and GPU respectively. Using separate backends makes it very easy to deploy PyTorch on constrained systems.

---

### Writing first neural network in pytorch

```
“Talk is cheap. Show me the code” 

- Linus Torvalds
```

For the sake of demonstration we will be designing a neural network in Pytorch with 1 10-neuron hidden layer, an input and an output layer. 

The reader can take the code below and try to add more layers and experiment with changing the width(number of neurons) in the hidden layers to see how it affects the output or the training process.

Some stuff to know before proceeding to the following code - 
1. Epoch - It is a measure of the number of times all of the training data are used once to update the weights. An epoch can have multiple iteration steps. Since whole data can be very big to load in memory, so we often load the data in batches and pass it through the neural network, such passes of mini-batches are known as iteration steps. 
2. Iris dataset -  Iris flower data set is a multivariate dataset which has 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor) and their petal length, sepal length, petal width and sepal width.

![alt text](https://github.com/vaibhawvipul/First-steps-towards-Deep-Learning/blob/master/Chapter-2/assests/Chapter-2-fig-1.png "Figure 1")

```
# Let us start with importing the libraries
# We will go in depth later

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# Taking a few data points from iris dataset . 
# Here X represents the list of [petal length, petal width and sepal length]. 
# Y is the sepal width. 

X = torch.tensor(([5.1, 3.5, 1.4], [6.7, 3.1, 4.4], [6.5, 3.2, 5.1]), dtype=torch.float) # 3 X 3 tensor
y = torch.tensor(([0.2], [1.4], [2]), dtype=torch.float) # 3 X 1 tensor
xTestInput = torch.tensor(([5.9, 3, 5.1]), dtype=torch.float)

# Given X we will try to predict sepal width using neural networks

# We will now define a neural network in pytorch
# It will have two fully connected layers.
# In nn.Linear(3,10) , 3 specifies the input size and 10 specifies the output size.
# we will be giving 3 data points to the neural network [petal length, petal width and sepal length] and it will have 10 neurons hidden layer which will pass the data to another layer fc2.
# fc2 will give us 1 output i.e sepal width
  
class Neural_Network(nn.Module):
    def __init__(self):
        super(Neural_Network, self).__init__()
        self.fc1 = nn.Linear(3, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x

# The following code is for training neural networks

model = Neural_Network()
model.train()

# we are using stochastic gradient descent optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# we train the neural networks for 1000 epochs.
# epochs basically means the number of time the whole set of data will pass through the neural network.

for i in range(1000):
    optimizer.zero_grad()
    output = model(X)

    # We calculate the loss, i.e difference between ground truth and predicted value.
    loss = F.binary_cross_entropy_with_logits(output, Y)
    print(loss)

    # Following code is to do backpropagation and for updating weights.   
    loss.backward()
    optimizer.step()

# Now our model is trained, so we pass a test input to see how the model is performing.
print(model(xTestInput))
```

---

### Essentials of pytorch

We have seen above an example which demonstrates how easy and intuitive it is to write. In this section we will try to dive deeper into Pytorch.

```
import torch 
import torch.nn as nn

```
torch.nn contains all the necessary tools we would need while coding up a neural network like Linear Layers, RNNs etc. Reader should check this link out in order to understand more about torch.nn - [here](https://pytorch.org/tutorials/beginner/nn_tutorial.html)

This is how you initialize a tensor in pytorch. In pytorch everything is a tensor.

```
x = torch.tensor(([5.1,4.3,2.5],[5.1,4.3,2.5],[5.1,4.3,2.5]), dtype=torch.float)
```

A Variable wraps a Tensor and supports nearly all the API’s defined by a Tensor. Variable also provides a backward method to perform backpropagation.

```
from torch.autograd import Variable
a = Variable(torch.Tensor([[1,2],[3,4]]), requires_grad=True)
b = torch.sum(a**2)

# compute gradients of b with respect to a
b.backward() 
print(a.grad())
```

One more thing which often comes handy is the knowledge of numpy to pytorch conversion and vice versa.

```
import numpy as np
a = np.array([1,2,3]) # numpy array
b = torch.from_numpy(a) # pytorch Tensor	

```

We will keep using pyTorch in the book, especially in chapter 3 and 4. Readers are advised to refer Pytorch’s documentation, it is one of the best resources out there. Also fast ai’s module which is based on pytorch is extremely beginner friendly. It is worth a try.

