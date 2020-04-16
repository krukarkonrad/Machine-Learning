import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import Sequential
from keras.optimizers import SGD

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

from keras.layers import Dropout, Dense

#set seed for reproduction purpose
from numpy.random import seed
seed(1)

# from tensorflow import
# set_random_seed(2)

import random as rn
rn.seed(12345)

import tensorflow as tf
tf.random.set_seed(1234)

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

images_train =  []
for image_train in x_train:
    images_train.append(image_train.flatten())

images_test = []

for image_test in x_test:
    images_test.append(image_test.flatten())

images_train = np.array(images_train)
images_test = np.array(images_test)

from keras.utils import normalize
images_train = normalize(images_train)
images_test = normalize(images_test)


from keras.utils.np_utils import to_categorical

y_train_categorical = to_categorical(y_train)
y_test_categorical = to_categorical(y_test)
"""
neural_network_mnist = Sequential()
neural_network_mnist.add(Dense(50, activation='relu', input_shape=(784,)))
#neural_network_mnist.add(Dense(20, activation='relu'))
#neural_network_mnist.add(Dropout(0.01))
neural_network_mnist.add(Dense(10, activation='softmax'))

neural_network_mnist.summary()

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9)
neural_network_mnist.compile(optimizer=sgd, loss="categorical_crossentropy", \
                             metrics=["accuracy"])

run_hist_3 = neural_network_mnist.fit(images_train, y_train_categorical, epochs=20, \
                                  validation_data=(images_test, y_test_categorical), \
                                  verbose=True, shuffle=False)

neural_network_mnist.save('mnist-model.h5')
"""
from keras.models import load_model
model = load_model('mnist-model.h5')
print("Model evaluation [loss, accu]: ", model.evaluate(images_test, y_test_categorical))

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

id = 150

print("Neural netowrk predicts:")
print(model.predict(images_test[id,:].reshape(-1,784)))

plt.imshow(images_test[id,:].reshape(28,28), cmap=plt.get_cmap("gray"))
plt.title('Image to recognize')

print("\nNeural network recognized image as:", np.argmax(model.predict(images_test[id,:].reshape(-1,784))))

plt.show()