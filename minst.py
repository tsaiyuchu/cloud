import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

# Loading data
mnist = tfds.load('mnist', split=None)
mnist_train = mnist['train']
mnist_test = mnist['test']

# Set algorithm parameters (hyperparameters)
epochs = 10
batch_size = 128
input_size = 28 * 28
output_size = 10
learning_rate = 0.001

# Initialize variables
weights = tf.Variable(tf.random.normal(shape=(input_size, output_size), dtype=tf.float32))
biases = tf.Variable(tf.random.normal(shape=(output_size,), dtype=tf.float32))
optimizer = tf.optimizers.Adam(learning_rate)

# Training loop
loss_values = []
epochs_list = []
for repeat in range(epochs):
    for batch in mnist_train.batch(batch_size, drop_remainder=True):
        # Transform and normalize data
        labels = tf.one_hot(batch['label'], output_size)
        X = batch['image']
        X = tf.cast(tf.reshape(X, [-1, input_size]), tf.float32) / 255.0

        with tf.GradientTape() as tape:
            # Define the model structure
            logits = tf.add(tf.matmul(X, weights), biases)
            # Declare the loss function
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels, logits))

        gradients = tape.gradient(loss, [weights, biases])
        optimizer.apply_gradients(zip(gradients, [weights, biases]))

    loss_values.append(loss.numpy())
    epochs_list.append(repeat + 1)
    print(f"epoch {repeat+1}, loss: {loss.numpy():.3f}")



# Evaluate the training model
test_batches = mnist_test.batch(batch_size)
preds = []
ground_truth = []
for batch in test_batches:
    X = tf.cast(tf.reshape(batch['image'], [-1, input_size]), tf.float32) / 255.0
    labels = tf.one_hot(batch['label'], output_size)
    batch_preds = tf.math.argmax(tf.add(tf.matmul(X, weights), biases), axis=1)
    batch_truth = tf.math.argmax(labels, axis=1)
    preds.append(batch_preds)
    ground_truth.append(batch_truth)

preds = np.concatenate(preds)
ground_truth = np.concatenate(ground_truth)

correct_preds = tf.reduce_sum(tf.cast(tf.equal(preds, ground_truth), tf.float32))
accuracy = correct_preds / len(mnist_test)

print(f"accuracy on test set: {accuracy.numpy():.3f}")

# Plot training loss history
plt.plot(epochs_list, loss_values, 'r')
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()