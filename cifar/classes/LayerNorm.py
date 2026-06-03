import numpy as np

from classes.autograd import Matrix


class LayerNorm:
    def __init__(self, hidden_size, eps=1e-5):

        self.eps = eps

        self.gamma = Matrix(np.ones((1, 1, hidden_size)))

        self.beta = Matrix(np.zeros((1, 1, hidden_size)))

    def __call__(self, x):

        mean = np.mean(x.matrix, axis=-1, keepdims=True)

        var = np.var(x.matrix, axis=-1, keepdims=True)

        x_hat = (x.matrix - mean) / np.sqrt(var + self.eps)

        out = Matrix(
            self.gamma.matrix * x_hat + self.beta.matrix, (x, self.gamma, self.beta)
        )

        def _backward():

            N = x.shape[-1]

            dy = out.grad * self.gamma.matrix

            std_inv = 1.0 / np.sqrt(var + self.eps)

            dvar = np.sum(
                dy * (x.matrix - mean) * -0.5 * std_inv**3, axis=-1, keepdims=True
            )

            dmean = np.sum(dy * -std_inv, axis=-1, keepdims=True) + dvar * np.mean(
                -2 * (x.matrix - mean), axis=-1, keepdims=True
            )

            dx = dy * std_inv + dvar * 2 * (x.matrix - mean) / N + dmean / N

            x.grad += dx

            self.gamma.grad += np.sum(out.grad * x_hat, axis=(0, 1), keepdims=True)

            self.beta.grad += np.sum(out.grad, axis=(0, 1), keepdims=True)

        out._backward = _backward

        return out

    def parameters(self):
        return [self.gamma, self.beta]