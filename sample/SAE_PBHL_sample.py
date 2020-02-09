import numpy as np
from DSAE_PBHL import SAE_PB
from tensorflow.keras.optimizers import Adam

data = np.load("datas/data.npy").astype(np.float32)

input_dim = data.shape[1]
hidden_dim = input_dim // 2

pb_input_dim = 4
pb_hidden_dim = 3

pb_idx = np.random.choice(pb_input_dim, size=data.shape[0])

pb = np.identity(pb_input_dim, dtype=np.float32)[pb_idx]

model = SAE_PB(
    input_dim, hidden_dim,
    pb_input_dim, pb_hidden_dim,
    alpha=0.003, beta=0.7, eta=0.5
)
optimizer = Adam()

epochs = 10000

for t in range(epochs):
    loss = model.train(optimizer, data, pb)
    print(f"epoch {t+1:06d}, loss {loss:.5f}")

np.save("datas/hidden.npy", model.encode(data))
