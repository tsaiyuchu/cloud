import numpy as np
import tensorflow as tf

# Load iris dataset
iris = np.load('train.npz')

# Convert to TensorFlow Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((iris['data'], iris['label']))
train_generator = train_dataset.shuffle(buffer_size=100).batch(20)  

# Define the multi-class model using a neural network with Dropout to prevent overfitting
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),  
    tf.keras.layers.Dropout(0.5),  
    tf.keras.layers.Dense(8, activation='relu'),  
    tf.keras.layers.Dropout(0.5),  
    tf.keras.layers.Dense(3)  
])

# Compile the model with the optimizer, loss function, and evaluation metric
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.02),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, epochs=300, verbose=1)  # Train for 1000 epochs

# Save the model to a .h5 file
model.save('IRIS.h5')
print("Model saved as 'IRIS.h5'")
