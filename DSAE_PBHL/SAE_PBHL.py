import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

class SAE_PBHL(tf.keras.Model):

    def __init__(self,
        input_size, hidden_size,
        pb_input_size, pb_hidden_size,
        alpha=0.003, beta=0.7, eta=0.5,
        activator=tf.nn.tanh, pb_activator=tf.nn.softmax
        ):
        super(SAE_PBHL, self).__init__()
        self.encode_layer = Dense(hidden_size, input_shape=(input_size, ), activation=activator)
        self.decode_layer = Dense(input_size, input_shape=(hidden_size+pb_hidden_size, ), activation=activator)
        self.pb_encode_layer = Dense(pb_hidden_size, input_shape=(input_size+pb_input_size, ), activation=activator)
        self.pb_decode_layer = Dense(pb_input_size, input_shape=(pb_hidden_size, ), activation=pb_activator)
        self.alpha = alpha
        self.beta = beta
        self.eta = eta

    def encode(self, X):
        return self.encode_layer(X)

    def encode_pb(self, X, pb_X):
        X_concat = tf.concat((X, pb_X), axis=1)
        return self.pb_encode_layer(X_concat)

    def decode(self, H, H_pb):
        H_concat = tf.concat((H, H_pb), axis=1)
        return self.decode_layer(H_concat)

    def decode_pb(self, H_pb):
        return self.pb_decode_layer(H_pb)

    @tf.function
    def calc_loss(self, X, X_pb):
        H = self.encode(X)
        H_pb = self.encode_pb(X, X_pb)
        X_reconst = self.decode(H, H_pb)
        X_pb_reconst = self.decode_pb(H_pb)

        reconstruction_loss = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(X - X_reconst), axis=1))
        reconstruction_loss += 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(X_pb - X_pb_reconst), axis=1))

        encoder_kernel = self.encode_layer.kernel
        decoder_kernel = self.decode_layer.kernel
        pb_encoder_kernel = self.pb_encode_layer.kernel
        pb_decoder_kernel = self.pb_decode_layer.kernel
        regularization_loss = 0.5 * (tf.reduce_sum(tf.square(encoder_kernel)) + tf.reduce_sum(tf.square(decoder_kernel)))
        regularization_loss += 0.5 * (tf.reduce_sum(tf.square(pb_encoder_kernel)) + tf.reduce_sum(tf.square(pb_decoder_kernel)))

        H_norm = 0.5 * (1.0 + tf.reduce_mean(H, axis=0))
        H_pb_norm = 0.5 * (1.0 + tf.reduce_mean(H_pb, axis=0))
        kl_loss = tf.reduce_sum(
            self.eta * tf.math.log(self.eta) -
            self.eta * tf.math.log(H_norm) +
            (1.0 - self.eta) * tf.math.log(1.0 - self.eta) -
            (1.0 - self.eta) * tf.math.log(1.0 - H_norm),
        )
        kl_loss += tf.reduce_sum(
            self.eta * tf.math.log(self.eta) -
            self.eta * tf.math.log(H_pb_norm) +
            (1.0 - self.eta) * tf.math.log(1.0 - self.eta) -
            (1.0 - self.eta) * tf.math.log(1.0 - H_pb_norm),
        )

        loss = reconstruction_loss + self.alpha * regularization_loss + self.beta * kl_loss
        return loss

    @tf.function
    def train(self, optimizer, X, X_pb, epoch=1):
        for _ in range(epoch):
            with tf.GradientTape() as tape:
                loss = self.calc_loss(X, X_pb)
            gradients = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss
