# Binary Classification using keras and Deep Learning

<p>
Keras is a Python library for deep learning that wraps the efficient numerical libraries TensorFlow and Theano. It allows you to
quickly design and train neural network and deep learning models. Here we will build a convolutional neural network to identify
images of dogs and cats. First the network will be trained on thousands of images, and it will be able to predict if a given test image is a cat or a dog.
  </p>
 <figure>
<img src="https://github.com/gereziherw/Binary-Classification-using-keras-and-Deep-Learning-/blob/master/images/Binary_Classifier.gif?raw=true">
  <figcaption>Fig1. Simple Binary CNN. </figcaption>
  </figure>
The process of building a Convolutional Neural Network always involves four major steps. 

<b>Step 1: 	Convolution</b>
Convolution is a weighted sum between two signals (in terms of signal processing jargon) or functions (in terms of mathematics). 
<figure>
<img src="https://github.com/gereziherw/Binary-Classification-using-keras-and-Deep-Learning-/blob/master/images/convolution-example-matrix.gif?raw=true">
  <figcaption>Fig2. Convolution</figcaption>
</figure>
<b>Step 2: 	Pooling </b>
pooling layers are used to reduce the size of image. It works by sampling in each layer using filters. 
Consider the following 4×4 layer. So if we use a 2×2 filter with stride 2 and max-pooling, 
we get the following response:
<figure>
<img src="https://github.com/gereziherw/Binary-Classification-using-keras-and-Deep-Learning-/blob/master/images/pooling.png?raw=true">
  <figcaption>Fig3. Max-Pooling </figcaption>
 </figure>
The Above Fig indicates a 4 2×2 matrix are combined into 1 and their maximum value is taken. Generally, max-pooling is 
used but other options like average pooling can be considered.

<b>Step 3: 	Flattening</b>
It Flattens the input without affecting the batch size.
<pre><i>Example
    model = Sequential()
    model.add(Conv2D(64, 3, 3,border_mode='same', input_shape=(3, 32, 32)))
    #now: model.output_shape == (None, 64, 32, 32)

    model.add(Flatten())
    #now: model.output_shape == (None, 65536)
</i></pre>

<b>Step 4: 	Full connection</b>
At the end of convolution and pooling layers, networks generally use fully-connected layers in which each pixel is
considered as a separate neuron just like a regular neural network. The last fully-connected layer will 
contain as many neurons as the number of classes to be predicted. For instance, if we have 10 classes, 
the last fully-connected layer will have 10 neurons.
for more detailed explanation about the above four steps plase visit click <a href = "https://www.analyticsvidhya.com/blog/2016/04/deep-learning-computer-vision-introduction-convolution-neural-networks/">here</a>

 <b>Structure of the Data set</b>
 First you need to collect your training and test data, and structure as follows. 
 
 <pre>
      CNN_Image_Classifier
      ||___ Dataset
          |___ Training_Set
            |___ Cats
              |___ cats_01.jpg
              |___ cats_02.jpg
              |___ ……. 
            |___ Dogs
              |___ dogs_01.jpg
              |___ dogs_02.jpg
              |___ ……. 
         |___ Test_Set
            |___ Cats
              |___ cats_01.jpg
              |___ cats_02.jpg
              |___ ……. 
           |___ Dogs
              |___ dogs_01.jpg
              |___ dogs_02.jpg
              |___ ……. 

         |___ Predict
            |___ cat_dog_01.jpg
             |___ ………
      ||___ train_CNN.py
      ||___ predict.py
      ||___ model.h5
      ||___ weights.h5

 </pre>

<b>How to run the code?</b>
<p>First download images of your own interest and put in the appropriate folder. Then, there are two ways to run.</p>
<ol>
  <li>Using the model and weights here: If you don’t want to spent more time to train your network, you can directly use the two .h5 files, and run predict to classify your image under predict. N.B. the model and weights given here are trained in 1500 cats and 1500 dogs of images and are validated with 1000 images of both cat and dog each with 500 images. The network has been trained with 5 epochs.</li>
  <li>
   If you want to build your own model, and train the network with your own images, then delete the two .h5 files and run the train_CNN.py file.  After it finishes the training two .h5 files will be created and finally run your predict.py code to make the prediction. 
  </li>
  <b>CNN for Multiclass classifier coming Soon </b>
