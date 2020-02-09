import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

class AE(tf.keras.Model):

    def __init__(self,
        input_size, hidden_size,
        activator=tf.nn.tanh
        ):
        super(AE, self).__init__()
        self.encode_layer = Dense(hidden_size, input_shape=(input_size, ), activation=activator)
        self.decode_layer = Dense(input_size, input_shape=(hidden_size, ), activation=activator)


    def encode(self, X):
        return self.encode_layer(X)

    def decode(self, H):
        return self.decode_layer(H)

    @tf.function
    def calc_loss(self, X):
        H = self.encode(X)
        X_reconst = self.decode(H)

        loss = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(X - X_reconst), axis=1))
        return loss

    @tf.function
    def train(self, optimizer, X, epoch=1):
        for _ in range(epoch):
            with tf.GradientTape() as tape:
                loss = self.calc_loss(X)
            gradients = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss
