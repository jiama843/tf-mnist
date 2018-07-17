# tf-mnist
Training models against the mnist dataset

# Notes


CNN Modelling
Convolutional Neural Nets for processing MNIST

----------
Description

Convolutional networks are a type of network that is primarily used to develop classification models.

They are in a class of feed-forward neural networks.


Components


1. Convolutional Layers 
![Convolutional Filter](https://www.researchgate.net/profile/Miguel_Nicolau/publication/308853748/figure/fig1/AS:413795764719617@1475668006140/Sample-generation-of-a-5-5-feature-map-as-the-result-of-the-2D-discrete-convolution-of.jpg)


A each layer represents a convolutional filter. For reach subregion of a matrix (could be a tensor), the convolutional layer produces a single value that is passed in as a feature to the next layer.


2. Pooling Layers


![Pooling Filter](https://upload.wikimedia.org/wikipedia/commons/thumb/e/e9/Max_pooling.png/314px-Max_pooling.png)


Each layer is used to down-sample the data. If a tensor was passed in as input to this layer, the corresponding output would be a reduced tensor.


![ReLU in processing pooling layer](https://upload.wikimedia.org/wikipedia/commons/d/dc/RoI_pooling_animated.gif)


Often times, a ReLU activation function is used to introduce non-linearity to the model. The function is in the form $$f(x) = max(0, x)$$ and is used on pooling sections.


3. Dense (Fully Connected) Layers

The last layers of this neural net. High-level reasoning on output processed by the convolutional and max pooling layers, all nodes are connected.


----------


Basic CNN Example

This is an analysis of the sample code provided on the tensorflow website.


1. Imports

The main imports are numpy (for data handling) and tensorflow (for the library that has a variety of support for ML). It is pretty much always necessary to set up a logger that can print information statements.


    from __future__ import absolute_import
    from __future__ import division
    from __future__ import print_function
    
    # Imports
    import numpy as np
    import tensorflow as tf
    
    tf.logging.set_verbosity(tf.logging.INFO)
    
    # Our application logic will be added here
    
    if __name__ == "__main__":
      tf.app.run()


2. 
