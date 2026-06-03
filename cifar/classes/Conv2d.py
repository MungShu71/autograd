import numpy as np

from classes.autograd import Matrix


class Conv2d:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        fan_in = in_channels * kernel_size * kernel_size

        scale = np.sqrt(2.0 / fan_in)

        self.w = Matrix(
            np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        )
        self.b = Matrix(np.zeros((out_channels,)))

    def __call__(self, x):

        B, C, H, W = x.shape

        K = self.kernel_size
        S = self.stride

        out_h = (H - K) // S + 1
        out_w = (W - K) // S + 1

        shape = (B, out_h, out_w, C, K, K)

        strides = (
            x.matrix.strides[0],
            S * x.matrix.strides[2],
            S * x.matrix.strides[3],
            x.matrix.strides[1],
            x.matrix.strides[2],
            x.matrix.strides[3],
        )

        patches = np.lib.stride_tricks.as_strided(
            x.matrix, shape=shape, strides=strides
        )

        x_col = patches.reshape(B * out_h * out_w, -1)

        w_col = self.w.matrix.reshape(self.out_channels, -1)

        out_matrix = (x_col @ w_col.T) + self.b.matrix

        out_matrix = out_matrix.reshape(B, out_h, out_w, self.out_channels)

        out_matrix = out_matrix.transpose(0, 3, 1, 2)

        out = Matrix(out_matrix, (x, self.w, self.b))

        def _backward():

            grad = out.grad.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)

            self.b.grad += np.sum(grad, axis=0)

            grad_w = grad.T @ x_col

            self.w.grad += grad_w.reshape(self.w.shape)

        out._backward = _backward

        return out

    def parameters(self):
        return [self.w, self.b]
