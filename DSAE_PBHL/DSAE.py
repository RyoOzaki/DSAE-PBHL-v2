import numpy as np
import tensorflow as tf
from .SAE import SAE

class DSAE(tf.keras.Model):

    def __init__(self,
        structure,
        **kwargs
        ):
        super(DSAE, self).__init__()
        networks = []
        for in_size, hidden_size in zip(structure[:-1], structure[1:]):
            networks.append(
                SAE(in_size, hidden_size, **kwargs)
            )
        self.networks = networks

    def encode_all(self, X):
        encoded = []
        for idx, net in enumerate(self.networks):
            X = net.encode(X)
            encoded.append(X)
        return encoded

    def encode(self, X):
        return self.encode_all(X)[-1]

    def decode_all(self, H):
        decoded = []
        for net in self.networks[::-1]:
            H = net.decode(H)
            decoded.append(H)
        return decoded

    def decode(self, H):
        return self.decode_all(H)[-1]

    @tf.function
    def train(self, optimizer, target_net_idx, X, epoch=1):
        target_net = self.networks[target_net_idx]
        for net in self.networks:
            if net == target_net:
                break
            X = net.encode(X)
        return target_net.train(optimizer, X, epoch=epoch)
