import numpy as np
import tensorflow as tf

class Cosine(tf.keras.layers.Layer):
    def __init__(self, units=1, input_dim=1):
        super().__init__()

    def call(self, inputs):
        return tf.math.cos(inputs)
