import numpy as np
from DSAE_PBHL import DSAE
from tensorflow.keras.optimizers import Adam

data = np.load("datas/data.npy").astype(np.float32)

input_dim = data.shape[1]

structure = [input_dim, 8, 5, 3]
model = DSAE_PBHL(structure, alpha=0.003, beta=0.7, eta=0.5)
network_N = len(model.networks)
optimizers = [Adam() for _ in range(network_N)]

epochs = 1000

print("============Start train")
for target_net_idx in range(network_N):
    for t in range(epochs):
        loss = model.train(
            optimizers[target_net_idx], target_net_idx, data, epoch=10
        )
        print(f"layer {target_net_idx+1}")
        print(f"epoch {t+1:06d}, loss {loss:.5f}")

np.save("datas/hidden.npy", model.encode(data))
