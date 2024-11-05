import os
import json
import tensorflow as tf

os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["localhost:12345", "localhost:23456", "localhost:34567"]
    },
    'task': {'type': 'worker', 'index': 2}  
})


strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))


with strategy.scope():

    model = tf.keras.Sequential([tf.keras.layers.Dense(10, activation='relu'),
                                  tf.keras.layers.Dense(1)])
    model.compile(optimizer='adam', loss='mse')

    import numpy as np
    x_train = np.random.random((1000, 32))
    y_train = np.random.random((1000, 1))

    model.fit(x_train, y_train, epochs=5)
