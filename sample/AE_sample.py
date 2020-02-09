import numpy as np
from DSAE_PBHL import AE
from tensorflow.keras.optimizers import Adam

data = np.load("datas/data.npy").astype(np.float32)

input_dim = data.shape[1]
hidden_dim = input_dim // 2

model = AE(input_dim, hidden_dim)
optimizer = Adam()

epochs = 10000

for t in range(epochs):
    loss = model.train(optimizer, data)
    print(f"epoch {t+1:06d}, loss {loss:.5f}")

np.save("datas/hidden.npy", model.encode(data))
