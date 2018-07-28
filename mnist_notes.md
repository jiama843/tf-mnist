# Convolutional Neural Nets for processing MNIST

Offical dropbox paper link: https://paper.dropbox.com/doc/Convolutional-Neural-Nets-for-processing-MNIST--AIENsPcRgP_xNCXg4kTgQO5qAQ-u4htcWO65ATgF3iO88p5R

----------
# Description

Convolutional networks are a type of network that is primarily used to develop classification models.

They are in a class of feed-forward neural networks.


![](https://www.kdnuggets.com/wp-content/uploads/convolutional-neural-net-process-3.png)



# Components
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


# Basic CNN Example

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


2. Input Layer (28x28x1)

We define our input layer as a 28x28x1 tensor of data.

This is because we flatten our input features, and then we generate a 28x28x1 tensor containing the results of the output.

The 28x28 represents the size of each matrix (feature) and the x1 represents the number of features.

→ Reshape

-1 denotes the batch size, which is currently undefined.


    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])


3. Convolutional Layer #1 (28x28x1 → 28x28x32)

The first convolutional layer applies 32 filters, resulting in 32 features.

The input tensor is subdivided into 5x5 subregions with stride = 1 and to compute the output tensor of one feature, of which there are 32.


     conv1 = tf.layers.conv2d(
          inputs=input_layer,
          filters=32,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu)

The conv2d function takes in an input tensor and a variety of other parameters to produce the input tensor for the next network layer (28x28x32). At the end we should have a size of 28x28 for each feature tensor. This is due to “same” padding.

→ Conv2D


    tf.layers.conv2d(inputs, filters, kernel_size, padding, activation)

The parameters are as follows:

- inputs -  The input tensor
- filters - specifies the number of filters applied (32)
- kernel_size - specifies the dimensions of the subregions (5x5)
- padding - specifics how to handle padding:
  - This picture is very clear in explaining the difference between same and valid 👌 :
    
![Difference between valid and same padding](https://d2mxuefqeaa7sj.cloudfront.net/s_F79CC2354D35F80AD8022015218E3EF5B9B8969BB014D80A367DAD7C70939A5A_1531845712696_image.png)

- activation - Specifies the type of activation function used (in this case and in the general case of CNN’s we use ReLU activation)


4. Pooling Layer #1 (28x28x32 → 14x14x32)

Using the output of Convolutional Layer #1 (conv1) as input, we run a pooling filter. We define the output as pool1. 

We use max_pooling, a method that involves taking the maximum value in each tensor subregion and passing it through to the next layer. 

In the example below, there is a pool size of 2x2, with size 2 strides. There are 4 subregions and it is clear that the maximum value in each of the subregions is used to create the next output layer.

![max pooling](https://upload.wikimedia.org/wikipedia/commons/thumb/e/e9/Max_pooling.png/314px-Max_pooling.png)



     pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

→ Max Pooling 2D


    tf.layers.max_pooling2d(inputs, pool_size, strides)

The parameters are as follows:

- inputs -  The input tensor (conv1)
- pool_size - specifies the dimensions of each subregion to determine the maximum of.
- strides - specifies the number of blocks to skip when filtering subregions.
  - The examples below use convolutional filters, however the concept of strides is still illustrated correctly 
![](http://machinelearninguru.com/_images/topics/computer_vision/basics/convolutional_layer_1/stride1.gif)
![](http://machinelearninguru.com/_images/topics/computer_vision/basics/convolutional_layer_1/stride2.gif)

5. Convolutional Layer #2 (14x14x32 → 14x14x64)

Same idea as the first convolutional layer, except we define 64 filters, so our output has 64 feature tensors.


    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)


6. Pooling Layer #2 (14x14x32 → 7x7x64)

Same idea as the first pooling layer, we deem this to be an effective downsampled size.


    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)


7. Dense Layer (7x7x64 → 1)

The steps to creating the dense layer (with ReLU activation) are as follows:

- Flatten the tensor that results from the 2nd pooling layer pool2. (7x7x64 → 3136)
- Use the flattened tensor pool2 as input to a dense layer function. (3136 → 1)
- Use the output of the dense layer dense as input to dropout function (dropout regularization)


    ```pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
      dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
      dropout = tf.layers.dropout(
          inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

→ Dropout regularization

A method where random nodes in a neural net are dropped during training steps in an effort to reduce overfitting of a dataset. 

In essence, we disable some nodes in the neural net while training so that the remaining (enabled) nodes can benefit. 
http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf



    tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    
    
The parameters are as follows:

- inputs -  The input tensor (in this case it is the result dense)
- rate - Rate of dropout (number of nodes to drop). e.g with drop rate 0.4, we are dropping 40% of nodes.
- training - boolean that determines whether to:
  - Return the output in training mode → apply dropout while training the model
  - Return the output in inference mode → apply dropout after model is trained (in a practical application, some nodes are turned off)

→ Dense layer


    tf.layers.dense(
        inputs,
        units,
        activation=None)


The parameters are as follows:

- inputs -  The input tensor
- units - Specifies the dimensionality of the output (number of output nodes)
- activation - The applied activation function (None specifies linear activation)


8. Logits layer (1024 → 10)

This layer returns a raw output using the results tensor dropout. The final dense layer, given 1024 input neurons will return one of 10 output neurons representing the possible numbers (0 - 9).

This concludes the creation of the model.


    tf.layers.dense(inputs=dropout, units=10)

After the creation of a model, important attributes such as weights need to be optimized in order for the model to attain maximal accuracy.

To train tensorflow models, there exists a class known as the estimator class.


Training the model

After the creation of a model, important attributes such as weights need to be optimized in order for the model to attain maximal accuracy.

To train tensorflow models, there exists a class known as the estimator class.

The estimator class encapsulates 4 main actions:

1. training
2. evaluation
3. prediction
4. export for serving

They are excellent since graph building, exception handling, creating checkpoint files etc. would be controlled effectively.

→ Loss

![A meme](https://i.kym-cdn.com/news/images/desktop/000/000/157/cca.png)


When prediction probability is not 1 (100%), there is a discrepancy between the predicted and observed values from a dataset. This can be measured in a quantity known as loss.

There are various types of methods for calculating loss, in this example, cross entropy loss is used. Cross entropy loss is widely regarded as the most important and most used loss function.

Some other types of loss:

- MSE (Mean Squared Error)
- L1 Loss (Linear Absolute value)
- L2 Loss (Based on L1)

In cross entropy loss function depicts a logarithmic relationship between loss (log loss) and predicted probability.


![Cross Entropy Loss](http://ml-cheatsheet.readthedocs.io/en/latest/_images/cross_entropy.png)


As the model gets more accurate in making predictions, the loss decreases logarithmically.


      # Calculate Loss (for both TRAIN and EVAL modes)
      loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

TensorFlow loss functions take in two parameters:

1. labels - These are the actual, observed values from the dataset
2. logits - These are the predicted values that result as output from the model (In ML, it has a different meaning than in the field of mathematics)

The cross entropy loss function would return the result: $$-\sum\limits_i p_i \log{(q_i)}$$ where $$p_{i=1} = y$$ and$$p_{i=0} = 1- y$$. and $$q$$ is the same, except with $$\hat{y}$$ replacing $$y$$.

$$y = \ label\ value$$ - predicted value (from model)
$$\hat{y} = \ logit\ value$$ - actual value

→ ModeKeys

ModeKeys are booleans defined in the estimator class that can be used in accordance with FLAGS ( https://stackoverflow.com/questions/45162446/whats-the-purpose-of-flags-in-tensorflow - after command line arguments are parsed, we determine which mode should be run)


1. EVAL
2. TRAIN
3. PREDICT

e.g:


    if mode == tf.estimator.ModeKeys.PREDICT:

mode is also used as a parameter to specify what fields need to be configured when calling EstimatorSpec

→ Configuring the estimator

In this section, we will analyze the configuration of the estimator below:


      if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
        
      # Configure the Training Op (for TRAIN mode)
      if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    
      # Add evaluation metrics (for EVAL mode)
      eval_metric_ops = {
          "accuracy": tf.metrics.accuracy(
              labels=labels, predictions=predictions["classes"])}
      return tf.estimator.EstimatorSpec(
          mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
          
EstimatorSpec is used to configure the fields.

Depending on the value of mode, different arguments are required:


1. For `mode == ModeKeys.TRAIN`: required fields are `loss` and `train_op`.


    tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

train_op is defined as the operation for the training step. For now, it is defined as  `tf.train.Optimizer.minimize`. 

This function is responsible for updating model weights and incrementing the global step - (a counter that keeps track of the current training step).


2. For `mode == ModeKeys.EVAL`: required field is `loss`.


    tf.estimator.EstimatorSpec(mode=mode, loss=loss)

Both of these fields have been explained in other sections of these notes.


3. For `mode == ModeKeys.PREDICT`: required fields are `predictions`


    tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

predictions is a tensor of predictions or a dictionary of predictions.

Overall, a model function can be configured with the template below so that it can be valid in the context of an estimator:


    def my_model_fn(mode, features, labels):
      if (mode == tf.estimator.ModeKeys.TRAIN or
          mode == tf.estimator.ModeKeys.EVAL):
        loss = ...
      else:
        loss = None
      if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = ...
      else:
        train_op = None
      if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = ...
      else:
        predictions = None
    
      return tf.estimator.EstimatorSpec(
          mode=mode,
          predictions=predictions,
          loss=loss,
          train_op=train_op)


Results of the sample example

