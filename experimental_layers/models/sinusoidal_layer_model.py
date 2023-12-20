import numpy as np
import tensorflow as tf
from experimental_layers.layers import Cosine

class SinusoidalLayerModel(tf.keras.Model):
    def __init__(self):
        super(SinusoidalLayerModel, self).__init__()
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, input_shape=(1,)),
            Cosine(1, 1),
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

    def get_angular_velocity(self):
        return self.model.layers[0].weights[0].numpy()[0][0]
    
    def get_phase(self):
        return self.model.layers[0].weights[1].numpy()[0]