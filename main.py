import tensorflow as tf
import numpy as np
from keras.datasets import cifar10
from keras.optimizers import SGD
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

class_names = np.array(['airplane','automobile ','bird ','cat ','deer ','dog ','frog ','horse ','ship ','truck'])

X_train = X_train / 255
X_test = X_test / 255

feathures_count = class_names.shape[0]

y_train = tf.keras.utils.to_categorical(y_train, feathures_count)
y_test = tf.keras.utils.to_categorical(y_test, feathures_count)

model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(32, 32, 3), name='Input_Layer'))
model.add(tf.keras.layers.Flatten(name='Flatten_Layer'))
model.add(tf.keras.layers.Dense(512, activation='relu', name='Hidden_Layer_1'))
model.add(tf.keras.layers.Dense(256, activation='relu', name='Hidden_Layer_2'))
model.add(tf.keras.layers.Dense(feathures_count, activation='softmax', name='Output_Layer'))
model.summary()

sgd = SGD(lr = 0.01)
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=64, epochs=10, validation_split=0.2)
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
print('Accuracy: %s' % accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))



model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(32, 32, 3), name='Input_Layer'))
model.add(tf.keras.layers.Conv2D(filters=10, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu', name='Unlinear_Layer_1'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'))
model.add(tf.keras.layers.Flatten(name='Flatten_Layer'))
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='relu', name='Unlinear_Layer_2'))
model.add(tf.keras.layers.Dense(feathures_count, activation='softmax', name='Output_Layer'))

sgd = SGD(lr = 0.01)
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

model.fit(X_train, y_train, batch_size=64, epochs=10, validation_split=0.2)
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
print('Accuracy: %s' % accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=128,
                           kernel_size=(2, 2),
                           padding='same',
                           activation='relu',
                           input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPool2D(pool_size=(3, 3), padding='valid'),
    tf.keras.layers.Conv2D(filters=128, 
                           kernel_size=(3, 3),
                           padding='same',
                           activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'),
    tf.keras.layers.Conv2D(filters=256, 
                           kernel_size=(3, 3),
                           padding='same',
                           activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, batch_size=64, epochs=10, validation_split=0.2)
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
print('Accuracy: %s' % accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))


import matplotlib.pyplot as plt

plt.figure(figsize=[30,20])
for i in range(48):
    plt.subplot(6, 8, i + 1)
    plt.xlabel(class_names[np.argmax(y_pred[i])])
    plt.imshow(X_test[i])

