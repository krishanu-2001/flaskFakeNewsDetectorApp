import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
#parameters
w = tf.Variable([.3],tf.float32)
b = tf.Variable([-.3],tf.float32)

# inputs and outputs
X = tf.placeholder(tf.float32)

linear_model = w*X + b
y = tf.placeholder(tf.float32)

#loss
squared_delta = tf.square(linear_model - y)

loss = tf.reduce_sum(squared_delta)

#optimize
optimizer =  tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)

for i in range(1000):
    sess.run(train,{X:[1,2,3,4],y:[0,-1,-2,-3]})

#print(sess.run(loss,{X:[1,2,3,4],y:[0,-1,-2,-3]}))

print(sess.run([w,b]))