import numpy as np
import tensorflow as tf

# Load test dataset
iris_test = np.load('test.npz')
test_data = tf.data.Dataset.from_tensor_slices((iris_test['data'], iris_test['label'])).batch(20)  

# Load trained model
model = tf.keras.models.load_model('IRIS.h5')

# Compile the model before evaluation
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.02),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_data, verbose=2)

# Print the test loss and accuracy
print(f"Test loss: {test_loss:.3f}")
print(f"Accuracy: {test_accuracy * 100:.2f}%")
