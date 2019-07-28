#!/usr/local/bin/python3

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

learning_rate = 0.01
epochs = 200


n_samples = 30 # we only need 30 samples for linear regression

train_x = np.linspace(0,20,n_samples)
train_y = 3*train_x + 4*np.random.randn(n_samples) # np.random.randn gives an array of random floats of size

# THIS IS IDEALLY WHAT THE BOUNDARY / REGRESSION LINE THAT FITS SHOULD LOOK LIKE y=mx + c , here m=3(weight) and c =0(bias)
plt.plot(train_x,train_y,'ro')
plt.plot(train_x,3*train_x,'b')
plt.show()


# CREATING TENSORS

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(np.random.randn(),name="weights")
B = tf.Variable(np.random.randn(),name="bias")

#Prediction
#pred = X*W + B.   or
pred = tf.add(tf.multiply(X,W), B)

#Cost Function
cost = tf.reduce_sum((pred - Y)**2) / (2*n_samples)

#Optimizer - Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()



with tf.Session() as sesh:
    sesh.run(init)
    
    for epoch in range(epochs):
        for x,y in zip(train_x,train_y):
            sesh.run(optimizer,feed_dict={X: x, Y: y})
            
        if not epoch%20:
            c= sesh.run(cost,feed_dict={X: train_x, Y: train_y})
            w= sesh.run(W)
            b= sesh.run(B)
            print(f'epoch:{epoch:04d} c:{c:.4f} w:{w:.4f} b:{b:.4f}')
        #Plotting the final stuff
    weight = sesh.run(W)
    bias = sesh.run(B)
    plt.plot(train_x,train_y,'ro')
    plt.plot(train_x, weight*train_x + bias)
    plt.show()