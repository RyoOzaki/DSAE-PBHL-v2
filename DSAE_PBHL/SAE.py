import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

class SAE(tf.keras.Model):

    def __init__(self,
        input_size, hidden_size,
        alpha=0.003, beta=0.7, eta=0.5,
        activator=tf.nn.tanh
        ):
        super(SAE, self).__init__()
        self.encode_layer = Dense(hidden_size, input_shape=(input_size, ), activation=activator)
        self.decode_layer = Dense(input_size, input_shape=(hidden_size, ), activation=activator)
        self.alpha = alpha
        self.beta = beta
        self.eta = eta

    def encode(self, X):
        return self.encode_layer(X)

    def decode(self, H):
        return self.decode_layer(H)

    @tf.function
    def calc_loss(self, X):
        H = self.encode(X)
        X_reconst = self.decode(H)

        reconstruction_loss = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(X - X_reconst), axis=1))

        encoder_kernel = self.encode_layer.kernel
        decoder_kernel = self.decode_layer.kernel
        regularization_loss = 0.5 * (tf.reduce_sum(tf.square(encoder_kernel)) + tf.reduce_sum(tf.square(decoder_kernel)))

        H_norm = 0.5 * (1.0 + tf.reduce_mean(H, axis=0))
        kl_loss = tf.reduce_sum(
            self.eta * tf.math.log(self.eta) -
            self.eta * tf.math.log(H_norm) +
            (1.0 - self.eta) * tf.math.log(1.0 - self.eta) -
            (1.0 - self.eta) * tf.math.log(1.0 - H_norm),
        )

        loss = reconstruction_loss + self.alpha * regularization_loss + self.beta * kl_loss
        return loss

    @tf.function
    def train(self, optimizer, X, epoch=1):
        for _ in range(epoch):
            with tf.GradientTape() as tape:
                loss = self.calc_loss(X)
            gradients = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss
