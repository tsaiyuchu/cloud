import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# Load iris dataset
iris = tfds.load('iris', split='train[:90%]', as_supervised=True)
iris_test = tfds.load('iris', split='train[90%:]', as_supervised=True)

batch_size = 20

# Data preprocessing
def iris2d(features, label):
    return features[2:], tf.cast((label == 0), dtype=tf.float32)

train_generator = iris.map(iris2d).shuffle(buffer_size=100).batch(batch_size)
test_generator = iris_test.map(iris2d).batch(1)

# Define the linear model
def linear_model(X, A, b):
    my_output = tf.add(tf.matmul(X, A), b)
    return tf.squeeze(my_output)

# Define SVM loss function
def svm_loss(A, y_true, y_pred):
    h = tf.keras.losses.Hinge()
    return tf.reduce_mean(tf.nn.l2_loss(A)/5.0 + h(y_true, y_pred))

my_opt = tf.optimizers.SGD(learning_rate=0.02)

# Training

tf.random.set_seed(1)
np.random.seed(0)

A = tf.Variable(tf.random.normal(shape=[2, 1]))
b = tf.Variable(tf.random.normal(shape=[1]))

history = list()

for i in range(300):
    iteration_loss = list()
    for features, label in train_generator:
        with tf.GradientTape() as tape:
            predictions = linear_model(features, A, b)
            loss = svm_loss(A, label, predictions)
        iteration_loss.append(loss.numpy())
        gradients = tape.gradient(loss, [A, b])
        my_opt.apply_gradients(zip(gradients, [A, b]))
    history.append(np.mean(iteration_loss))
    if (i + 1) % 30 == 0:
        print(f'Step # {i+1} Weights: {A.numpy().T} Biases: {b.numpy()}')
        print(f'Loss = {loss.numpy()}')

# Plot training loss history
plt.plot(history)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()

# Evaluate on test data
predictions = list()
labels = list()
for features, label in test_generator:
    predictions.append(linear_model(features, A, b).numpy())
    labels.append(label.numpy()[0])

test_loss = svm_loss(A, labels, predictions).numpy()
print(f"test loss is {test_loss}")

coefficients = np.ravel(A.numpy())
intercept = b.numpy()

# Plotting batches of examples
for j, (features, label) in enumerate(train_generator):
    setosa_mask = label.numpy() == 1
    setosa = features.numpy()[setosa_mask]
    non_setosa = features.numpy()[~setosa_mask]
    plt.scatter(setosa[:,0], setosa[:,1], c='red', label='setosa')
    plt.scatter(non_setosa[:,0], non_setosa[:,1], c='blue', label='Non-setosa')
    if j==0:
        plt.legend(loc='lower right')

# Compute and plot decision function
a = -coefficients[0] / coefficients[1]
xx = np.linspace(plt.xlim()[0], plt.xlim()[1], num=10000)
yy = a * xx - intercept / coefficients[1]
on_the_plot = (yy > plt.ylim()[0]) & (yy < plt.ylim()[1])
plt.plot(xx[on_the_plot], yy[on_the_plot], 'k--')

plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.show()