import tensorflow_datasets as tfds
import numpy as np

batch_size = 20

# train:test=8:2
iris_train = tfds.load('iris', split='train[:80%]', as_supervised=True) 
iris_test = tfds.load('iris', split='train[80%:]', as_supervised=True)


def save_to_npz(data, filename):
    features = []
    labels = []
    for feature, label in data:
        features.append(feature.numpy())
        labels.append(label.numpy())
    np.savez(filename, data=np.array(features), label=np.array(labels))

# Save datasets to .npz files
save_to_npz(iris_train, 'train.npz')
save_to_npz(iris_test, 'test.npz')

print("Datasets saved as train.npz and test.npz")
