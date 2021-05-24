import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, losses
import numpy as np
(x_train, y_train), (x_test, y_test)=tf.keras.datasets.mnist.load_data()
x_train = tf.expand_dims(x_train, axis=3, name=None)
x_test = tf.expand_dims(x_test, axis=3, name=None)

path = '/home/thlarsen/ood_detection/bayes_mnist/point_lenet_save/'

def train_model():

    model = models.Sequential()
    model.add(layers.Conv2D(6, kernel_size=5, padding='SAME', activation='relu', input_shape=x_train.shape[1:]))
    model.add(layers.MaxPooling2D(2,padding='SAME'))
    # model.add(layers.Activation('sigmoid'))
    model.add(layers.Conv2D(16, 5, padding='SAME', activation='relu'))
    model.add(layers.MaxPooling2D(2, padding='SAME'))
    # model.add(layers.Activation('sigmoid'))
    model.add(layers.Conv2D(120, 5, padding='SAME', activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(84, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.summary()

    model.build(input_shape=[None, 28, 28, 1])


    model.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])

    history = model.fit(x_train, y_train, batch_size=64, epochs=40, validation_data=(x_test, y_test))

    model.evaluate(x_test, y_test)

    model.save(path)

def load_model(): 
    return tf.keras.models.load_model(path)

def main(): 
    TRAIN = False 

    if TRAIN: 
        train_model()
    else:
        model = load_model() 
        print(model.layers)
        # for l in model.layers: 
        l = model.layers[0]
        weights = l.get_weights()
        W = np.array(weights[0])
        b = weights[1]
        print(W.shape)
        print(b.shape)


if __name__ == '__main__':
    main() 


