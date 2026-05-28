import time
import numpy as np

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

# =========================================================
# DATA
# =========================================================


class Dataloader:
    def __init__(self, image_file, label_file):
        self.images = np.load(image_file).transpose(0, 3, 1, 2).astype(np.float32)
        self.labels = np.load(label_file)

        self.num_samples = self.images.shape[0]

    def draw(self, index=0):
        image = self.images.reshape(-1, 3 * 32 * 32)[index].astype(int)
        for y in range(0, 32, 2):
            for x in range(32):
                idx = x + y * 32
                idx2 = x + (y + 1) * 32
                r1, g1, b1 = image[idx], image[idx + 32 * 32], image[idx + 32 * 32 * 2]
                r2, g2, b2 = (
                    image[idx2],
                    image[idx2 + 32 * 32],
                    image[idx2 + 32 * 32 * 2],
                )
                print(f"\x1b[38;2;{r1};{g1};{b1}m\x1b[48;2;{r2};{g2};{b2}m ", end="")
            print("\x1b[0m")

        print(f"Label: {classes[self.labels[index]]}")
        print(f"Label: {[self.labels[index]]}")

    def preprocess(self, start, batch_size):
        x = self.images[start : start + batch_size] / 255.0
        y = self.labels[start : start + batch_size].astype(np.int32)

        one_hot = np.zeros((len(y), 10), dtype=np.float32)
        one_hot[np.arange(len(y)), y] = 1.0

        return x, y, one_hot


# =========================================================
# AUTOGRAD
# =========================================================


class Matrix:
    def __init__(self, matrix, _children=()):
        self.matrix = np.array(matrix, dtype=np.float32)
        self.grad = np.zeros_like(self.matrix)

        self._prev = set(_children)
        self._backward = lambda: None

        self.shape = self.matrix.shape

    # -----------------------------------------------------

    def __repr__(self):
        return f"Matrix({self.matrix.flatten()} | {self.grad.flatten()})"


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


# =========================================================
# LAYERS
# =========================================================


class Linear:
    def __init__(self, in_dim, out_dim):

        scale = np.sqrt(2.0 / in_dim)

        self.w = Matrix(np.random.randn(in_dim, out_dim) * scale)

        self.b = Matrix(np.zeros((1, out_dim)))

    def __call__(self, x):

        return x.matmul(self.w) + self.b

    def parameters(self):
        return [self.w, self.b]


# =========================================================


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


# =========================================================


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
        print(self.w.shape)
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


# =========================================================
# TRANSFORMER
# =========================================================


class FeedForward:
    def __init__(self, hidden_size, intermediate_size):

        self.fc1 = Linear(hidden_size, intermediate_size)
        self.fc2 = Linear(intermediate_size, hidden_size)

    def __call__(self, x):

        x = self.fc1(x).gelu()

        x = self.fc2(x)

        return x

    def parameters(self):

        return self.fc1.parameters() + self.fc2.parameters()


# =========================================================


class MultiHeadAttention:
    def __init__(self, hidden_size, num_heads):

        self.hidden_size = hidden_size

        self.num_heads = num_heads

        self.d_head = hidden_size // num_heads

        self.query = Linear(hidden_size, hidden_size)
        self.key = Linear(hidden_size, hidden_size)
        self.value = Linear(hidden_size, hidden_size)

        self.proj = Linear(hidden_size, hidden_size)

    def __call__(self, x):

        B, N, _ = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = q.reshape(B, N, self.num_heads, self.d_head).transpose(0, 2, 1, 3)

        k = k.reshape(B, N, self.num_heads, self.d_head).transpose(0, 2, 1, 3)

        v = v.reshape(B, N, self.num_heads, self.d_head).transpose(0, 2, 1, 3)

        attn = q.matmul(k.transpose(0, 1, 3, 2))

        attn = attn * (1 / np.sqrt(self.d_head))
        
        print(attn.shape)

        attn = attn.softmax()

        out = attn.matmul(v)

        out = out.transpose(0, 2, 1, 3)

        out = out.reshape(B, N, self.hidden_size)

        out = self.proj(out)

        return out

    def parameters(self):

        return (
            self.query.parameters()
            + self.key.parameters()
            + self.value.parameters()
            + self.proj.parameters()
        )


# =========================================================


class Block:
    def __init__(self, hidden_size, intermediate_size, num_heads):

        self.norm1 = LayerNorm(hidden_size)

        self.attn = MultiHeadAttention(hidden_size, num_heads)

        self.norm2 = LayerNorm(hidden_size)

        self.ffn = FeedForward(hidden_size, intermediate_size)

    def __call__(self, x):

        x = x + self.attn(self.norm1(x))

        x = x + self.ffn(self.norm2(x))

        return x

    def parameters(self):

        return (
            self.norm1.parameters()
            + self.attn.parameters()
            + self.norm2.parameters()
            + self.ffn.parameters()
        )


# =========================================================


class Encoder:
    def __init__(self, depth, hidden_size, intermediate_size, num_heads):

        self.blocks = [
            Block(hidden_size, intermediate_size, num_heads) for _ in range(depth)
        ]

    def __call__(self, x):

        for block in self.blocks:
            x = block(x)

        return x

    def parameters(self):

        params = []

        for block in self.blocks:
            params.extend(block.parameters())

        return params


# =========================================================
# EMBEDDING
# =========================================================


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


# =========================================================
# VIT
# =========================================================


class VIT:
    def __init__(
        self,
        image_size,
        patch_size,
        in_channels,
        hidden_size,
        intermediate_size,
        num_heads,
        depth,
        num_classes,
    ):

        self.embedder = Embedder(image_size, patch_size, in_channels, hidden_size)

        self.encoder = Encoder(depth, hidden_size, intermediate_size, num_heads)

        self.norm = LayerNorm(hidden_size)

        self.classifier = Linear(hidden_size, num_classes)

    def forward(self, x):

        x = self.embedder(x)

        x = self.encoder(x)

        x = self.norm(x)

        cls = x[:, 0, :]

        logits = self.classifier(cls)

        return logits

    def parameters(self):

        return (
            self.embedder.parameters()
            + self.encoder.parameters()
            + self.norm.parameters()
            + self.classifier.parameters()
        )


# =========================================================
# LOSS
# =========================================================


class CrossEntropyLoss:
    def __call__(self, logits, labels):

        probs = logits.softmax(axis=1)

        epsilon = 1e-12

        log_probs = np.log(probs.matrix + epsilon)

        loss_value = -np.sum(labels.matrix * log_probs) / probs.shape[0]

        out = Matrix(loss_value, (logits,))

        def _backward():

            logits.grad += (probs.matrix - labels.matrix) / probs.shape[0]

        out._backward = _backward

        return out, probs


# =========================================================
# OPTIMIZER
# =========================================================


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


# =========================================================
# TRAINER
# =========================================================


class Trainer:
    def __init__(self, model, optimizer, criterion, train_data, eval_data, batch_size):

        self.model = model

        self.optimizer = optimizer

        self.criterion = criterion

        self.train_data = train_data

        self.eval_data = eval_data

        self.batch_size = batch_size

    def train(self, epochs):

        num_samples = self.train_data.num_samples

        for epoch in range(epochs):

            start = time.perf_counter()

            total_loss = 0
            total_correct = 0

            indices = np.random.permutation(num_samples)

            self.train_data.images = self.train_data.images[indices]
            self.train_data.labels = self.train_data.labels[indices]

            for i in range(0, num_samples, self.batch_size):

                self.optimizer.zero_grad()

                x, y, one_hot = self.train_data.preprocess(i, self.batch_size)

                X = Matrix(x)

                Y = Matrix(one_hot)

                logits = self.model.forward(X)

                loss, probs = self.criterion(logits, Y)

                total_loss += loss.matrix

                preds = np.argmax(probs.matrix, axis=1)

                total_correct += np.sum(preds == y)

                loss.backward()

                # gradient clipping
                max_norm = 1.0

                total_norm = np.sqrt(
                    sum(np.sum(p.grad**2) for p in self.optimizer.params)
                )

                if total_norm > max_norm:

                    coef = max_norm / (total_norm + 1e-6)

                    for p in self.optimizer.params:
                        p.grad *= coef

                self.optimizer.step()

            avg_loss = total_loss / (num_samples / self.batch_size)

            avg_acc = (total_correct / num_samples) * 100

            print(
                f"Epoch {epoch+1} | "
                f"Loss {avg_loss:.4f} | "
                f"Acc {avg_acc:.2f}% | "
                f"Time {time.perf_counter()-start:.1f}s"
            )

    def evaluate(self):

        num_samples = self.eval_data.num_samples

        total_correct = 0

        for i in range(0, num_samples, self.batch_size):

            x, y, _ = self.eval_data.preprocess(i, self.batch_size)

            X = Matrix(x)

            logits = self.model.forward(X)

            preds = np.argmax(logits.matrix, axis=1)

            total_correct += np.sum(preds == y)

        acc = (total_correct / num_samples) * 100

        print(f"Eval Accuracy: {acc:.2f}%")


# =========================================================
# MAIN
# =========================================================


def main():

    train_data = Dataloader("train_images.npy", "train_labels.npy")

    eval_data = Dataloader("test_images.npy", "test_labels.npy")

    vit = VIT(
        image_size=32,
        patch_size=4,
        in_channels=3,
        hidden_size=48,
        intermediate_size=48 * 4,
        num_heads=4,
        depth=3,
        num_classes=10,
    )

    optimizer = Adam(vit.parameters(), lr=3e-4)

    criterion = CrossEntropyLoss()

    trainer = Trainer(
        model=vit,
        optimizer=optimizer,
        criterion=criterion,
        train_data=train_data,
        eval_data=eval_data,
        batch_size=32,
    )

    trainer.train(epochs=10)

    trainer.evaluate()


if __name__ == "__main__":
    main()
    # a = Matrix([[3.1, 1.05]])
    # w = Matrix([[0.6], [0.7]])
    # b = Matrix([[0.9]])

    # x = a.matmul(w)
    # y = x + b # (1, 1)
    # z = y ** 2 # (1, 1)

    # z.backward()

    # print(f"a: {a}")
    # print(f"w: {w}")
    # print(f"b: {b}")
    # print(f"z: {z}")
