# Chapter 1
## What are artificial neural networks? What are its components?

### Inspiration 

As we human beings evolved with time, curiosity to understand certain subjects increased exponentially. Universe, Singularity, Meaning of life, God, Infinity and Brain were the ones which made to the top of the list. With time our brain became more and more efficient, we starting thinking about subjects deeply and we started asking the right questions. I will list a few of the beautiful questions down here, and while you read them take a pause and applaud the centuries of human progress it shows - 

- Are we alone in the universe?
- What is consciousness? Are all living beings conscious?
- What makes us human? - It can’t just be DNA because human genome is 99% identical to a chimpanzee
- What’s so weird about prime numbers? 
- Do we have free will?
- P versus NP 
- How does our brain work? 

Did you notice? How beautifully and precise are these questions? Generations of people have spent time in thinking through these problems. We came up with questions like these, we thought! We think! 

You see what I did there? Yes, Human Brain is the core of all the curiosity here. So what if we focus on developing a human brain? We will be able to put that brain to focus on trivial issues like driving a car,making sure that fraud doesn’t happen over credit card transactions etc. We can also put it to remote places where accessibility is an issue and solve complicated problems. We can put that brain to evaluate cardiovascular diseases via a retina scan. Now to do this, to carry out the tasks mentioned above we need intelligence. We need Artificial Intelligence!

Note - The [Blue Brain project](https://www.epfl.ch/research/domains/bluebrain/blue-brain/about/) is a dedicated team of folks who are working to build a digital reconsruction and simulations of a rodent and eventually human brain.

How do we develop AI? Where do we go for inspiration? When we had to build a plane, we studied birds. For building helicopters, dragonflies came for rescue. So to build a system which can be put to solve trivial as well as complicated problems, we need to look for inspiration from the most intelligent species on earth. We need to look inside a living being which has the largest cerebral cortex relative to their size. We would need to look inside Human Brain! 

--- 

### What are neural networks?

![alt text](https://github.com/vaibhawvipul/First-steps-towards-Deep-Learning/blob/master/Chapter-1/assests/Chapter-1-fig-1%20.png "Figure 1")


Before this representation of neural network starts unsettling us let us take time to understand how this architecture was thought of and how to interpret the image.

As mentioned in the section above, the neural networks are inspired by the way our brain works. The picture above represents a graphical flow in which the circles, represents neurons and the edges represent axon. The numbers { w1, w2 … wn } are set of weights of the neural networks. The weights are the key part of neural networks because these are the parameters which gets tuned when neural network goes under training process.  

Another thing to remember is that neural networks are not an exact representation of the brain. Brain is a very complicated organ and it is obvious that we don’t learn by the method of backpropagation(to be discussed later). Having said that let us further examine the picture above, if you observe carefully you will see that not all neurons in layer 1 are connected to all the neurons in layer 2. However, all the neurons in layer 2 is connected to all the neurons in layer 3(the final layer). This type of layer is known as a **Fully Connected Layer**. If a neural network has all layers as fully connected layers then that neural network is called **Fully Connected Network**(FCN). 

Before we digress to our next topic, let us spend some more time in understanding neural networks and their importance. 
Firstly, Neural networks are really powerful because of their ability to learn the relationships in a set of data on their own. Neural networks do not need to be told which features are important or not. Neural nets are capable of extracting the needed features and carry on the task efficiently.

Before neural nets became popular, people used to hardcode features by hand in the traditional machine learning algorithms. For example, in computer vision HOG(Histogram of Oriented Gradients) were used. Let us say that we want to detect a face then a hardcoded feature can be that there will always be a slope from cheeks to eyes. 

Following is an opencv code which can generate a HOG from an image for us.

```
from skimage import exposure,feature, data
import cv2
 
image = data.astronaut()
_ , hogImage = feature.hog(image, orientations=8, pixels_per_cell=(16, 16), 
	cells_per_block=(1, 1), transform_sqrt=True, 
	block_norm="L1", visualize=True, multichannel=True)
hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
hogImage = hogImage.astype("uint8")

while 1: 
	cv2.imshow("HOG Image", hogImage)
	cv2.waitKey(1)
```
We won’t go through the code for now. Following will be the output.

![alt text](https://github.com/vaibhawvipul/First-steps-towards-Deep-Learning/blob/master/Chapter-1/assests/Chapter-1-hog.png "HOG")

The challenge is that it is hard to write every possible feature in all the varying conditions by hand. That is why we give neural nets diverse data points so that it can learn these features by itself.  Neural networks adapt to the new inputs they see. If designed well then it has an amazing capability to generalize across various scenarios and often across domains. A neural network which is trained to differentiate between cats and dogs can also be used to differentiate between elephants and giraffes via the process of *transfer learning*.  

--- 

### Fully Connected Neural Networks

History is witness that the field of Artificial Intelligence has gone through multiple phases of boom-and-bust cycles. It is like stock markets, after every bull run there is a correction and once in a while there is recession and depression. 

I would say that the field of deep learning started with emergence of Multi-Layer Perceptrons(MLP). Simply put, it was a feed forward neural network. These MLPs had an input layer, a hidden layer and an output layer. MLPs could work well with data that is not linearly separable. The key thing about MLP is that all of its layers were fully connected. It means that each neuron of one layer was connected to all the neurons in the next layer.

![alt text](https://github.com/vaibhawvipul/First-steps-towards-Deep-Learning/blob/master/Chapter-1/assests/Chapter-1-fcn.png "FCN")

In the year 1975, backpropagation became the practical way to train MLP. However deep learning was still not a mainstream way to do machine learning. That was because MLPs were usually trained on XOR datasets. More than the lack of computational power there was also a lack of real life datasets on which the practicality of MLP could be verified. Around 1995, Yann LeCun released MNIST dataset. This dataset had collection of handwritten digits. This for some time became benchmark for forthcoming neural networks. 

In today’s era we have a multitude of datasets available. Imagenets for benchmarking object classification neural networks. MOT17, PASCAL VOC etc for object tracking and detection. However, MNIST dataset is still important today. It helps us in quickly prototyping and testing our models on a light-weight dataset. If a model is performing well on MNIST dataset then it is safe to pursue the design of the model one has thought of and then train it with heavy datasets. There are some datasets which can be useful along with MNIST i.e Fashion MNIST, Omniglot dataset, etc.

We have got a neural network which is fully connected. It has got one or more than one hidden layers between input and output layers. We have datasets as well. Now, let us dive deeper into training process of fully connected neural networks.

---

### BackPropagation

In the sections above we have seen that neural networks have neurons in the layers and every neuron may or may not be connected to all the neurons in the next layer. The concept to which one should pay attention here is that every connection from one neuron to another carries weight. 

Forward propagation is the process in which the input passes through the neural network to give an output(say probability). 
Backward propagation is the process in which we calculate how far we are from the ground truth and based on that we adjust the weights of the connections, so that next time we are closer to the ground truth. Intuitively, it tells every layer/connection/neuron that given the current input how much they were responsible for the wrong output and how should they correct themselves. In this way when a new input comes, we get closer to the ground truth and the loss is minimized.

Broadly speaking, there are three steps in training - 

1. Model Initialization (we will talk about it later)
2. Forward Propagation - This predict an output.
3. Back Propagation - On the basis of a defined loss function we calculate how far is the model from the ground truth and then we update the weights in the network. These updates happen via differentiation which the Optimizers do.

![alt text](https://github.com/vaibhawvipul/First-steps-towards-Deep-Learning/blob/master/Chapter-1/assests/Chapter-1-backprop.png "backprop")

Backpropagation is mathematically hard to understand. Luckily we have got deep learning frameworks like pytorch, tensorflow, darknet, Mxnet etc which takes care of automatic differentiation for us.

Backpropagation is just another name for automatic differentiation. Our agenda here is to build basic intuition that can get you started in the field of deep learning. If you are interested in knowing more about backpropagation and you want to dive deep into the mathematics(beyond the scope of this book) of it, I would suggest you to do a literature survey and read classic papers by Hinton and Yan LeCun.

Note - Geoffery Hinton, father of current AI boom, is deeply suspicious about Backpropagation.  He says “My view is throw it all away and start again, I don't think it's how the brain works. We clearly don't need all the labeled data". We humans try to find patterns in everything. We do it with the even scarcity and sparsity of data. There has to be a better way than backpropagation. We need to think more about unsupervised learning.

--- 

### Universal Approximators

Deep fully connected neural networks are often called as universal approximators. This is because no matter how complicated a function may look like, neural networks can approximate it.

![alt text](https://github.com/vaibhawvipul/First-steps-towards-Deep-Learning/blob/master/Chapter-1/assests/Chapter-1-univapprox.png "Universal Approximator")

Universal approximations are quite a common phenomena in mathematics as well as in computer science. 

A couple of things to keep in mind about universal approximation theorem is that neural networks can approximate a function and not exactly compute a function. Secondly, the function which is being approximated should be continuous. A function is said to be continuous for which sufficiently small changes in the input have arbitrarily small changes in the output. In simpler words, Neural networks can approximate sin(x) but not 1/|x| where x belongs to the set of real numbers. If you can formulate a problem well enough, neural networks can solve them for you. The part of problem formulation is difficult.

Universal Approximation theorem makes neural network truly remarkable. It allows us to work with any arbitrary functions. Imagine that all the data which you have collected is now plotted on a cartesian plane, there will be a wiggly line joining all those data points. Neural networks have this amazing capability to extract right features and predict based on those features or patterns.

If you are a mathematician or are interested in the proof of why Universal Approximation Theorem(beyond the scope of this book) holds for neural networks then you should read some original papers. One needs to be aware of fourier transformations, Hahn-Banach theorem etc to follow the proof.

---

### The more layers the better the network?

This is an active area of research and AI practitioners often find it hard to take one side. While designing a neural network it is often advised that it is better to add more layers rather than adding more neurons in a layer.

Basically, as the hypothesis space of a learning algorithm grows, the algorithm can learn richer and better relationships. However, chances of becoming prone to overfitting and its generalization error increases.
Hence, it is theoretically advisable to work with the minimal model that has enough capacity to learn the real structure of the data. But this is a very hand-wavy advice, because usually the core internal structure of the data is unknown, and often we don’t understand the models which we train. 

Shallow networks are very good at memorization, but not so good at *generalization*. The advantage of multiple layers is that they can learn features at various levels of abstraction. If one builds a very wide, very deep network, there are high chances of each layer just memorizing the output, and the neural network fails to generalize.

I know this is confusing! So one take on this can be to train a deep model and then try to minimize it. By following this approach one can quickly prototype and eventually follow *neural network compression and quantization*.

There is a dedicated research area for this topic called Neural Architecture Search (NAS), that focuses on creating algorithms or methods for finding the optimal architecture that fits certain data, by architecture meaning the number of layers, nodes, etc of a network. This is a subfield of AutoMachine Learning that it is growing in importance through this last years. More info and literature about it can be found ![here](https://www.automl.org/automl/literature-on-neural-architecture-search/)

---

### Types of Learning

Before moving forward, It is important to understand the various ways in which a neural network learns.

1. Supervised Learning - In this way of training a model, we provide huge amount of labelled dataset to the neural network. The dataset are in pairs of input and the desired output.
2. Unsupervised Learning - It is inspired by one of the oldest learning algorithms, Hebbian learning. In this way of learning a model is given a dataset which is neither labelled nor classified. 
3. Imitation Learning - This is popular in the field of reinforcement learning. In this mode of learning a policy is formed based on demonstrations.
4. Active Learning - In this mode of learning, the algorithm queries the source of information to get the desired output at new data points. 

We will be mostly dealing with *supervised learning* in this book. However, I encourage the reader to explore active learning on its own. It has got a lot of applications in the field of *Natural Language Processing (NLP)*. Computer Vision world still relies heavily on supervised learning. One of the reasons being huge availability of labelled datasets. However, even in computer vision, one shot learning and zero shot learning  have become active areas of research. Reader is advised to explore *ESZSL(Embarrassingly Simple Approach to Zero Shot Learning)* algorithm. 

---

### Weight Initialization

The sections above have acquainted us with the overall idea of training a neural network. One thing to understand here is how do we process the first input during training process? Recall that connections between various neurons have weights. How should we initialize those weights efficiently so that the learning can happen smoothly?
Let us say that we initialized all the weights as zero. If we do that then we will lose the symmetry inside the neural networks. During backpropagation every layer will have similar weight updates and every layer will learn the same thing.  This makes your model equivalent to a linear model.

Another way to solve the problem of weight initialization can be to randomly initialize the set of weights. This method often leads to two potential issues - 

1. Vanishing Gradient Problem - Simply put, during backpropagation the weight updates will be so less that neural network will stop learning totally no matter how many epochs you run it for or how much more data you feed it. 
2. Exploding gradients - This is opposite of vanishing gradient problem. It will stop model from reaching of global minima and during the optimization(of loss) the model will keep oscillating and will never learn.

There are a few methods in which we can overcome these problems like using relu activation function, dropouts or gradient clipping etc. 

Note - Pytorch, by default, uses *Xavier weight Initialization*. *He Initialization*, *Fixup Initialization* are some other ways in which weights in the neural network can be initialized however, those are beyond the scope of the book. 

--- 

### Activation functions

Neural networks like MLP were inspired by brains. Apart from having neurons, weights and connections they also have something similar to action potential known as Activation Functions. 

The idea is that we want an on/off mechanism for neurons. We want neurons to fire only when an input hits a certain threshold. Activation functions helps in introducing non-linearity into the output of a neuron. Non-linearity is an important property to have in neural networks because without non-linearity a neural network will be like a regression model.

There are a lot of activation functions available like sigmoid, tanh and relu (maxout and leaky relu are some variants of relu). Relu helps the neural networks architect to deal with vanishing gradient/gradient explosion problem. 

It is a very simple function - 
`f(x) = max(0,x)`
Basically, Relu outputs the maximum of zero or x.

Note - Whenever in doubt, use RELU(Rectified linear unit). 

--- 

### Understanding overfitting, underfitting and generalization

Keeping track of unending jargon is one of the toughest hurdles for a newbie entering in the field of Artificial Intelligence. It is important to understand Overfitting, Underfitting, and bias-variance trade-off because these are core terminologies of Machine Learning world.

One can say that the sole purpose of neural networks is to generalize well. Normally, we are used to algorithms like mergesort which are trained to do one particular task really well. In the machine learning world we want a model to detect object really well while also being able to do human pose estimation(Mask RCNN).

Overfitting happens when the model is performing really well on the training set, i.e the loss during training reduces but the model fails to perform well in test dataset/real world. While Underfitting happens when the model fails to learn from the training data and is unreliable.
A generalized model is neither overfit nor underfit. 

Note - An integral part of machine learning is bias-variance tradeoff. We can measure the generality of a model using this concept. Generalization is bound by the two undesirable outcomes — high bias and high variance. 

---

### Dropouts

Using dropouts can be an effective way to avoid overfitting of the model. 

The idea here is to switch off some neurons in layers of neural networks randomly during each iteration, so that not all the neurons see all the training data. By using dropouts we train an ensemble of neural networks rather than training a single architecture.

![alt text](https://github.com/vaibhawvipul/First-steps-towards-Deep-Learning/blob/master/Chapter-1/assests/Chapter-1-dropouts.png "Dropouts")

Looking at the figure above, we can see that the dropout probability for the second layer was 0.33 . It means that each neuron has 0.33% chances of getting “dropped out” or switched off. Let us say that the first neuron of the second layer was switched off, now when a data will pass through this network and weights will be updated via back-propagation then the weights of the connections connecting that neuron won’t be updated. It is like that the neuron never saw that data point, in next iteration another set of neurons will be switched off. 

Dropouts, however, can make us lose information. To counter this fact, Batch Normalization came into the picture. We can use less dropouts when we use Batch Normalization because it helps us to retain information while solving the problems of overfitting.

Dropouts are mostly used during training and deep learning frameworks make managing dropouts extremely easy. 

Note - For serious readers, I’d highly recommend reading the paper on dropouts by G.Hinton et.al.
