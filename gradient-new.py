import tensorflow as tf
import matplotlib.pyplot as mlp
import random

random.seed(0)

step = 20
rows = 100
slope = 0.4
bias = 1.5


x_train = tf.random.uniform(shape=(rows,)) 
noise = tf.random.normal(shape=(len(x_train),), stddev=0.01)
y_train = slope * x_train + bias + noise  

y_train_min = tf.reduce_min(y_train)
y_train_max = tf.reduce_max(y_train)
y_train = (y_train - y_train_min) / (y_train_max - y_train_min)


m = tf.Variable(0.)
b = tf.Variable(0.)


def predict_y_value(x):
    return m * x + b  


def squared_error(y_pred, y_true):
    return tf.reduce_mean(tf.square(y_pred - y_true))

loss = squared_error(predict_y_value(x_train), y_train)
print("Initial loss:", loss.numpy())

learning_rate = 0.05
steps = 2000


for i in range(steps):
    with tf.GradientTape() as tape:
        predictions = predict_y_value(x_train)  
        loss = squared_error(predictions, y_train)  
        gradients = tape.gradient(loss, [m, b]) 
        m.assign_sub(gradients[0] * learning_rate)
        b.assign_sub(gradients[1] * learning_rate)
        if (i % step) == 0:
            print("Step %d, Loss %f" % (i, loss.numpy()))


print("m: %f, b: %f" % (m.numpy(), b.numpy()))


mlp.scatter(x_train, y_train, color='blue', label='Training Data')
mlp.plot(x_train,predict_y_value(x_train), color='red', label='Predicted Price')
mlp.xlabel('x_train')
mlp.ylabel('y_train')
mlp.legend()
mlp.show()
