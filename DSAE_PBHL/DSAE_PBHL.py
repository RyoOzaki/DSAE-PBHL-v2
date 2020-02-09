import numpy as np
import tensorflow as tf
from .SAE import SAE
from .SAE_PBHL import SAE_PBHL

class DSAE_PBHL(tf.keras.Model):

    def __init__(self,
        structure,
        pb_structure,
        **kwargs
        ):
        super(DSAE_PBHL, self).__init__()
        sub_structure = structure[:-1]
        networks = []
        for in_size, hidden_size in zip(sub_structure[:-1], sub_structure[1:]):
            networks.append(
                SAE(in_size, hidden_size, **kwargs)
            )
        networks.append(
            SAE_PBHL(
                structure[-2], structure[-1],
                pb_structure[0], pb_structure[1],
                alpha=0.003, beta=0.7, eta=0.5
            )
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

    def encode_pb(self, X, X_pb):
        for net in self.networks[:-1]:
            X = net.encode(X)
        return self.networks[-1].encode_pb(X, X_pb)

    def decode_all(self, H, H_pb):
        decoded = []
        H = self.networks[-1].decode(H, H_pb)
        decoded.append(H)
        for net in self.networks[:-1][::-1]:
            H = net.decode(H)
            decoded.append(H)
        return decoded

    def decode(self, H, H_pb):
        return self.decode_all(H)[-1]

    def decode_pb(self, H_pb):
        return self.networks[-1].decode_pb(H_pb)

    @tf.function
    def train(self, optimizer, target_net_idx, X, X_pb=None, epoch=1):
        target_net = self.networks[target_net_idx]
        for net in self.networks:
            if net == target_net:
                break
            X = net.encode(X)
        if type(target_net) == SAE_PBHL:
            return target_net.train(optimizer, X, X_pb, epoch=epoch)
        else:
            return target_net.train(optimizer, X, epoch=epoch)
