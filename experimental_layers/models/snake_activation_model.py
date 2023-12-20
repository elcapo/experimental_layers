import numpy as np
import tensorflow as tf
from experimental_layers import layers
from experimental_layers.activations import snake

class SnakeActivationModel(tf.keras.Model):
    def __init__(self):
        super(SnakeActivationModel, self).__init__()
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(8, input_shape=(1,), activation=snake),
            tf.keras.layers.Dense(1),
        ])
    
    def call(self, input_tensor):
        return self.model(input_tensor)

    def train(self, x_train, y_train, epochs=300, learning_rate=0.1):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.MeanAbsoluteError(),
            metrics=['accuracy'],
        )
        return self.model.fit(x_train, y_train, epochs=epochs)