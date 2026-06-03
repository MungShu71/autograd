import numpy as np

from classes.Conv2d import Conv2d
from classes.autograd import Matrix


class Embedder:
    def __init__(self, image_size, patch_size, in_channels, hidden_size):

        self.num_patches = (image_size // patch_size) ** 2

        self.hidden_size = hidden_size

        self.projection = Conv2d(
            in_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )

        self.cls = Matrix(np.random.randn(1, 1, hidden_size) * 0.02)

        self.pos = Matrix(np.random.randn(1, self.num_patches + 1, hidden_size) * 0.02)

    def __call__(self, x):

        B = x.shape[0]

        z = self.projection(x)

        z = z.reshape(B, self.hidden_size, self.num_patches)

        z = z.transpose(0, 2, 1)

        cls = Matrix(np.tile(self.cls.matrix, (B, 1, 1)), (self.cls,))

        def _backward():
            self.cls.grad += np.sum(cls.grad, axis=0, keepdims=True)

        cls._backward = _backward

        z = Matrix.concatenate([cls, z], axis=1)

        z = z + self.pos

        return z

    def parameters(self):

        return self.projection.parameters() + [self.cls, self.pos]