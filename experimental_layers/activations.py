import tensorflow as tf

def snake(x):
    return x + tf.math.sin(x)**2

def snake_activation(x):
    return tf.keras.layers.Activation(snake)