import numpy as np

from classes.autograd import Matrix


class Linear:
    def __init__(self, in_dim, out_dim):

        scale = np.sqrt(2.0 / in_dim)

        self.w = Matrix(np.random.randn(in_dim, out_dim) * scale)

        self.b = Matrix(np.zeros((1, out_dim)))

    def __call__(self, x):

        return x.matmul(self.w) + self.b

    def parameters(self):
        return [self.w, self.b]