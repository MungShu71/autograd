import numpy as np


class Adam:
    def __init__(
        self, params, lr=3e-4, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01
    ):

        self.params = params

        self.lr = lr

        self.beta1 = beta1
        self.beta2 = beta2

        self.eps = eps

        self.weight_decay = weight_decay

        self.t = 0

        self.m = [np.zeros_like(p.matrix) for p in params]

        self.v = [np.zeros_like(p.matrix) for p in params]

    def zero_grad(self):

        for p in self.params:
            p.grad = np.zeros_like(p.grad)

    def step(self):

        self.t += 1

        for i, p in enumerate(self.params):

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad

            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (p.grad**2)

            m_hat = self.m[i] / (1 - self.beta1**self.t)

            v_hat = self.v[i] / (1 - self.beta2**self.t)

            update = m_hat / (np.sqrt(v_hat) + self.eps)

            update += self.weight_decay * p.matrix

            p.matrix -= self.lr * update