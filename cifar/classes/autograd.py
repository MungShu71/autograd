import numpy as np
class Matrix:
    def __init__(self, matrix, _children=()):
        self.matrix = np.array(matrix, dtype=np.float32)
        self.grad = np.zeros_like(self.matrix)

        self._prev = set(_children)
        self._backward = lambda: None

        self.shape = self.matrix.shape

    # -----------------------------------------------------

    def __repr__(self):
        return f"Matrix({self.matrix.flatten()} | Grad : {self.grad.flatten()})"


    # -----------------------------------------------------

    def backward(self):
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)

                for child in v._prev:
                    build(child)

                topo.append(v)

        build(self)

        self.grad = np.ones_like(self.matrix)

        for v in reversed(topo):
            v._backward()

    # -----------------------------------------------------

    def reshape(self, *shape):
        out = Matrix(self.matrix.reshape(*shape), (self,))

        def _backward():
            self.grad += out.grad.reshape(self.shape)

        out._backward = _backward
        return out

    # -----------------------------------------------------

    def transpose(self, *axes):
        if not axes:
            axes = tuple(range(len(self.shape))[::-1])

        out = Matrix(self.matrix.transpose(axes), (self,))

        def _backward():
            inverse = np.argsort(axes)
            self.grad += out.grad.transpose(inverse)

        out._backward = _backward
        return out

    # -----------------------------------------------------

    def __getitem__(self, item):
        out = Matrix(self.matrix[item], (self,))

        def _backward():
            grad = np.zeros_like(self.matrix)
            grad[item] = out.grad
            self.grad += grad

        out._backward = _backward
        return out

    # -----------------------------------------------------

    @staticmethod
    def reduce_grad(grad, shape):
        while len(grad.shape) > len(shape):
            grad = grad.sum(axis=0)

        for i, dim in enumerate(shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)

        return grad

    # -----------------------------------------------------

    def __add__(self, other):
        other = other if isinstance(other, Matrix) else Matrix(other)

        out = Matrix(self.matrix + other.matrix, (self, other))

        def _backward():
            self.grad += Matrix.reduce_grad(out.grad, self.shape)
            other.grad += Matrix.reduce_grad(out.grad, other.shape)

        out._backward = _backward
        return out

    # -----------------------------------------------------

    def __mul__(self, other):

        # scalar multiply
        if isinstance(other, (int, float)):
            out = Matrix(self.matrix * other, (self,))

            def _backward():
                self.grad += out.grad * other

            out._backward = _backward
            return out

        # elementwise multiply
        other = other if isinstance(other, Matrix) else Matrix(other)

        out = Matrix(self.matrix * other.matrix, (self, other))

        def _backward():
            self.grad += Matrix.reduce_grad(out.grad * other.matrix, self.shape)

            other.grad += Matrix.reduce_grad(out.grad * self.matrix, other.shape)

        out._backward = _backward
        return out

    # -----------------------------------------------------

    def matmul(self, other):
        other = other if isinstance(other, Matrix) else Matrix(other)

        out = Matrix(self.matrix @ other.matrix, (self, other))

        def _backward():

            grad_a = out.grad @ np.swapaxes(other.matrix, -1, -2)
            grad_b = np.swapaxes(self.matrix, -1, -2) @ out.grad

            self.grad += Matrix.reduce_grad(grad_a, self.shape)
            other.grad += Matrix.reduce_grad(grad_b, other.shape)

        out._backward = _backward
        return out

    # -----------------------------------------------------

    def __pow__(self, power):
        out = Matrix(self.matrix**power, (self,))

        def _backward():
            self.grad += (power * (self.matrix ** (power - 1))) * out.grad

        out._backward = _backward
        return out

    # -----------------------------------------------------
    
    def relu(self):
        x = self.matrix
        
        # Forward pass: replace all negative values with 0
        out_matrix = np.maximum(0, x)
        
        # Create the output Matrix and pass self as a child to track the graph
        out = Matrix(out_matrix, (self,))

        def _backward():
            # Derivative of ReLU is 1 if x > 0 else 0
            relu_grad = (x > 0).astype(np.float32)
            
            # Chain rule: multiply the incoming gradient by ReLU's derivative
            self.grad += relu_grad * out.grad

        out._backward = _backward
        return out
        

    def gelu(self):

        x = self.matrix

        tanh_term = np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))

        out_matrix = 0.5 * x * (1 + tanh_term)

        out = Matrix(out_matrix, (self,))

        def _backward():

            sech2 = 1 - tanh_term**2

            grad = 0.5 * (
                1
                + tanh_term
                + x * sech2 * np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * x**2)
            )

            self.grad += grad * out.grad

        out._backward = _backward

        return out

    # -----------------------------------------------------

    def softmax(self, axis=-1):

        x = self.matrix

        x = x - np.max(x, axis=axis, keepdims=True)

        exps = np.exp(x)

        probs = exps / np.sum(exps, axis=axis, keepdims=True)

        out = Matrix(probs, (self,))

        def _backward():

            s = out.matrix

            dot = np.sum(out.grad * s, axis=axis, keepdims=True)

            self.grad += s * (out.grad - dot)

        out._backward = _backward

        return out

    # -----------------------------------------------------
    def sum(self, axis=None, keepdims=False):
        out_matrix = np.sum(self.matrix, axis=axis, keepdims=keepdims)
        out = Matrix(out_matrix, (self, ))

        def _backwarad():
            if not keepdims and axis is not None:
                axes = [axis] if isinstance(axis, int) else axis
                shape = list(self.shape)
                for ax in axes:
                    shape[ax]=1
                grad_reshape = out.grad.reshape(shape)
            else:
                grad_reshape = out.grad
            self.grad += np.ones_like(self.matrix) * grad_reshape
        out._backward = _backwarad
        return out

    @staticmethod
    def concatenate(matrices, axis=0):

        raw = [m.matrix for m in matrices]

        out = Matrix(np.concatenate(raw, axis=axis), tuple(matrices))

        def _backward():

            sizes = [m.shape[axis] for m in matrices]

            indices = np.cumsum(sizes)[:-1]

            grads = np.split(out.grad, indices, axis=axis)

            for g, m in zip(grads, matrices):
                m.grad += g

        out._backward = _backward

        return out