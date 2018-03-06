# Binary-Classification-using-keras-and-Deep-Learning-
<pre>
<p>
Keras is a Python library for deep learning that wraps the efficient numerical libraries TensorFlow and Theano. It allows you to quickly design and train neural network and deep learning models. Here we will build a convolutional neural network to identify images of dogs and cats. First the network will be trained on thousands of images, and it will be able to predict if a given test image is a cat or a dog.
  </p>
The process of building a Convolutional Neural Network always involves four major steps. 
<b>Step 1: 	Convolution</b>
Convolution is a weighted sum between two signals (in terms of signal processing jargon) or functions (in terms of mathematics). 
(pic)
<b>Step 2: 	Pooling </b>
pooling layers are used to reduce the size of image. It works by sampling in each layer using filters. Consider the following 4×4 layer. So if we use a 2×2 filter with stride 2 and max-pooling, we get the following response:
<b>Step 3: 	Flattening</b>
It Flattens the input without affecting the batch size.
<p><i>Example
model = Sequential()
model.add(Conv2D(64, 3, 3,border_mode='same', input_shape=(3, 32, 32)))
# now: model.output_shape == (None, 64, 32, 32)

model.add(Flatten())
# now: model.output_shape == (None, 65536)
</i></b>
<b>Step 4: 	Full connection</b>
At the end of convolution and pooling layers, networks generally use fully-connected layers in which each pixel is considered as a separate neuron just like a regular neural network. The last fully-connected layer will contain as many neurons as the number of classes to be predicted. For instance, if we have 10 classes, the last fully-connected layer will have 10 neurons.
for more detailed explanation about the above four steps plase visit click <a href = "https://www.analyticsvidhya.com/blog/2016/04/deep-learning-computer-vision-introduction-convolution-neural-networks/">here</a>

</pre>
