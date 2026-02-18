import numpy as np

class Autoencoder:
    def __init__(self, input_dim, hidden_dim, learning_rate=0.01, seed=42):
        # Use ONE RNG everywhere (init + shuffling) for full reproducibility
        self.rng = np.random.default_rng(seed)

        # Xavier-ish init (small, stable)
        self.W_e = self.rng.normal(0, 1.0 / np.sqrt(input_dim), size=(hidden_dim, input_dim))
        self.b_e = np.zeros((hidden_dim, 1))

        self.W_d = self.rng.normal(0, 1.0 / np.sqrt(hidden_dim), size=(input_dim, hidden_dim))
        self.b_d = np.zeros((input_dim, 1))

        self.lr = learning_rate

    # ---------- helpers ----------
    @staticmethod
    def sigmoid(a):
        a = np.clip(a, -50, 50)  # avoid overflow
        return 1.0 / (1.0 + np.exp(-a))

    @staticmethod
    def sigmoid_prime(sigmoid_output):
        # derivative using already-computed sigmoid output
        return sigmoid_output * (1.0 - sigmoid_output)

    # ---------- forward ----------
    def encoder(self, x):
        """
        x: shape (input_dim, batch_size)
        returns z: shape (hidden_dim, batch_size)
        """
        self.s_e = self.W_e @ x + self.b_e
        z = self.sigmoid(self.s_e)
        return z

    def decoder(self, z):
        """
        z: shape (hidden_dim, batch_size)
        returns x_hat: shape (input_dim, batch_size)
        """
        self.s_d = self.W_d @ z + self.b_d
        x_hat = self.sigmoid(self.s_d)
        return x_hat

    # ---------- loss ----------
    def compute_loss(self, x, x_hat):
        m = x.shape[1]   # batch size
        loss = np.sum((x_hat - x)**2) / m
        return loss

    # ---------- backward ----------
    def backward(self, x, z, x_hat):
        """
        returns grads dict
        """
        batch_size = x.shape[1]

        # dL/dx_hat
        m = x.shape[1]
        d_xhat = 2 * (x_hat - x) / m  # (input_dim, B)

        # through sigmoid at decoder output
        d_sd = d_xhat * self.sigmoid_prime(x_hat)  # (input_dim, B)

        dW_d = d_sd @ z.T                           # (input_dim, hidden_dim)
        db_d = np.sum(d_sd, axis=1, keepdims=True)  # (input_dim, 1)

        # backprop to z
        d_z = self.W_d.T @ d_sd                     # (hidden_dim, B)

        # through sigmoid at encoder output
        d_se = d_z * self.sigmoid_prime(z)          # (hidden_dim, B)

        dW_e = d_se @ x.T                           # (hidden_dim, input_dim)
        db_e = np.sum(d_se, axis=1, keepdims=True)  # (hidden_dim, 1)

        return {"dW_e": dW_e, "db_e": db_e, "dW_d": dW_d, "db_d": db_d}

    # ---------- update ----------
    def step(self, grads):
        self.W_e -= self.lr * grads["dW_e"]
        self.b_e -= self.lr * grads["db_e"]
        self.W_d -= self.lr * grads["dW_d"]
        self.b_d -= self.lr * grads["db_d"]

    # ---------- training ----------
    def train(self, X, epochs=20, batch_size=128, shuffle=True, verbose=True):
        """
        X: shape (input_dim, N)
        returns list of epoch losses
        """
        N = X.shape[1]
        losses = []

        for ep in range(1, epochs + 1):
            if shuffle:
                perm = self.rng.permutation(N)
                X_ep = X[:, perm]
            else:
                X_ep = X

            epoch_loss = 0.0
            num_batches = 0

            for i in range(0, N, batch_size):
                xb = X_ep[:, i:i + batch_size]  # (input_dim, B)

                z = self.encoder(xb)
                x_hat = self.decoder(z)

                loss = self.compute_loss(xb, x_hat)
                grads = self.backward(xb, z, x_hat)
                self.step(grads)

                epoch_loss += loss
                num_batches += 1

            epoch_loss /= max(num_batches, 1)
            losses.append(epoch_loss)

            if verbose:
                print(f"Epoch {ep:02d}/{epochs} - loss: {epoch_loss:.6f}")

        return losses
