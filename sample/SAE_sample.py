import numpy as np
from DSAE_PBHL import SAE
from tensorflow.keras.optimizers import Adam

data = np.load("datas/data.npy").astype(np.float32)

input_dim = data.shape[1]
hidden_dim = input_dim // 2

model = SAE(input_dim, hidden_dim, alpha=0.003, beta=0.7, eta=0.5)
optimizer = Adam()

epochs = 100

print("============Start train")
for t in range(epochs):
    loss = model.train(optimizer, data, epoch=10)
    print(f"epoch {t+1:06d}, loss {loss:.5f}")

np.save("datas/hidden.npy", model.encode(data))
