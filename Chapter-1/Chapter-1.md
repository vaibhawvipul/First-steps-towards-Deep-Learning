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

