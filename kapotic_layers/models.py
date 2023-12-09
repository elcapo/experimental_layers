import numpy as np
import tensorflow as tf
from kapotic_layers import layers

class SinusoidalRegressionModel(tf.keras.Model):
    def __init__(self):
        super(SinusoidalRegressionModel, self).__init__()
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, input_shape=(1,)),
            layers.Cosine(1, 1),
        ])
    
    def call(self, input_tensor):
        return self.model(input_tensor)

def train_sinusoidal_regression_model(x_train, y_train, epochs=2000, learning_rate=0.01):
    model = SinusoidalRegressionModel()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.MeanAbsoluteError(),
    )
    model.fit(x_train, y_train, epochs=epochs)
    return model