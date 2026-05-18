import json

import numpy as np


class Dataloader:
    def __init__(self, image_file, label_file):
        self.images = np.fromfile(image_file, dtype=np.float32).reshape(-1, 28*28)
        self.labels = np.fromfile(label_file, dtype=np.float32)

        self.num_samples = self.images.shape[0]
    def draw(self, index=0):
        image = self.images[index]
        for y in range(28):
            for x in range(28):
                num = image[x+y*28]
                col = 232 + int(num * 23)
                print(f"\x1b[48;5;{col}m  ", end="")
            print("\x1b[0m")
        print("\x1b[0m")
        print("Label: ", end='')
        for i in range(10):
            print(1 if i==self.labels[index] else 0, end=' ')
        print()

    def preprocess(self, start, batch_size):
        batch_images = self.images[start:start+batch_size]
        batch_labels = self.labels[start:start+batch_size].astype(int)

        one_hot = np.zeros((len(batch_labels), 10))
        one_hot[np.arange(len(batch_labels)), batch_labels] = 1

        return batch_images, batch_labels, one_hot


class Linear:
    def __init__(self, in_dim, out_dim):
       
        scale = np.sqrt(2.0 / in_dim)
        self.w = Matrix(np.random.randn(in_dim, out_dim) * scale)
        self.b = Matrix(np.zeros((1, out_dim)))

    def __call__(self, x):
        return x * self.w + self.b

    def parameters(self):
        return [self.w, self.b]
    
    
class SGD:
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def step(self):
        for p in self.params:
            p.matrix -= self.lr * p.grad

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.grad)

class Adam:
    def __init__(self, params, lr, beta1=0.9, beta2=0.999, eps=1e-8, decay=1e-2) -> None:
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.decay = decay
        self.t = 0
        self.m = [np.zeros_like(p.matrix) for p in self.params]
        self.v = [np.zeros_like(p.matrix) for p in self.params]
    def step(self):

        self.t += 1
        for i, p in enumerate(self.params):
            self.m[i] = self.beta1 * self.m[i] + (1-self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1-self.beta2) * (p.grad ** 2)

            m_hat = self.m[i] / (1-self.beta1 ** self.t)
            v_hat = self.v[i] / (1-self.beta2 ** self.t)
            p.matrix -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.decay * p.matrix)
            # p.matrix -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) )

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.grad)


    
class CrossEntropyLoss:
    def __call__(self, logits, label):

        exps = np.exp(
            logits.matrix - np.max(logits.matrix, axis=1, keepdims=True)
        )

        probs = Matrix(
            exps / np.sum(exps, axis=1, keepdims=True),
            (logits,)
        )

        epsilon = 1e-12

        log_probs = np.log(probs.matrix + epsilon)

        batch_loss = (
            -np.sum(label.matrix * log_probs)
            / probs.matrix.shape[0]
        )

        out = Matrix(batch_loss, (probs, label))

        def _backward():

            logits.grad += (
                (probs.matrix - label.matrix)
                / probs.matrix.shape[0]
            )

        out._backward = _backward

        return out, probs


class Matrix:

    def __init__(self, matrix, _children=()):
        self.matrix = np.array(matrix)
        self.grad = np.zeros_like(self.matrix)
        self._prev = set(_children)
        self._backward = lambda: None
        self.shape = self.matrix.shape


    # def __add__(self, other):
    #     other = other if isinstance(other, Matrix) else Matrix(other)
    #     out = Matrix(self.matrix + other.matrix, (self, other))


    #     def _backward():
    #         self.grad += out.grad                        # pass full grad to left operand
    #         other.grad += np.sum(out.grad, axis=0, keepdims=True)  # sum over batch for bias
    #     out._backward = _backward            
    #     return out
    
    def __add__(self, other):
        other = other if isinstance(other, Matrix) else Matrix(other)
        out = Matrix(self.matrix + other.matrix, (self, other))


        def _backward():
            self.grad += out.grad                        # pass full grad to left operand
            axes_to_sum = tuple(range(out.grad.ndim - 1))
            # other.grad += np.sum(out.grad, axis=axes_to_sum, keepdims=True)
            other.grad += np.sum(out.grad, axis=tuple(range(out.grad.ndim - 1)))
        out._backward = _backward            
        return out
    # def __mul__(self, other):
    #     other = other if isinstance(other, Matrix) else Matrix(other)

    #     out = Matrix(self.matrix@other.matrix, (self, other))

    #     def _backward():
    #         self.grad += out.grad @ other.matrix.T
    #         other.grad += self.matrix.T @ out.grad
    #     out._backward = _backward            
    #     return out
    def __mul__(self, other):
        other = other if isinstance(other, Matrix) else Matrix(other)
        out = Matrix(self.matrix @ other.matrix, (self, other))

        def _backward():

            b_t = np.swapaxes(other.matrix, -1, -2)
            self.grad += out.grad @ b_t

            a_t = np.swapaxes(self.matrix, -1, -2)
            grad_for_b = a_t @ out.grad
            
            # Logic: If b is 2D (Weights) and a/grad are 3D (Data), we SUM the batch
            if other.matrix.ndim == 2 and grad_for_b.ndim > 2:
                other.grad += np.sum(grad_for_b, axis=tuple(range(grad_for_b.ndim - 2)))
            else:
                other.grad += grad_for_b
                
        out._backward = _backward            
        return out

    def __pow__(self, other):
        out = Matrix(self.matrix**other, (self,))


        def _backward():
            self.grad += (other * (self.matrix**(other-1))) * out.grad
        out._backward = _backward            

        return out

    def relu(self):
        out = Matrix(np.maximum(0, self.matrix), (self,))

        def _backward():
            self.grad += (self.matrix > 0) * out.grad                 
        out._backward = _backward

        return out

    def softmax(self, target):
        exps = np.exp(self.matrix - np.max(self.matrix, axis=-1, keepdims=True))
        probs = exps / np.sum(exps, axis=-1, keepdims=True)

        out = Matrix(probs, (self,))

        def _backward():
            pass 

        out._backward = _backward
        return out
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = np.ones_like(self.matrix)
        for v in reversed(topo):
            v._backward()

    def __repr__(self) -> str:
        return f"Matrix: {self.matrix} | Grad: {self.grad.flatten()}"

def avgpool2x2(x):

    batch = x.shape[0]

    images = x.matrix.reshape(batch, 28, 28)

    pooled = (
        images[:, 0::2, 0::2] +
        images[:, 1::2, 0::2] +
        images[:, 0::2, 1::2] +
        images[:, 1::2, 1::2]
    ) / 4.0

    out = Matrix(
        pooled.reshape(batch, 14 * 14),
        (x,)
    )

    def _backward():

        grad = out.grad.reshape(batch, 14, 14) / 4.0

        expanded = np.zeros((batch, 28, 28))

        expanded[:, 0::2, 0::2] += grad
        expanded[:, 1::2, 0::2] += grad
        expanded[:, 0::2, 1::2] += grad
        expanded[:, 1::2, 1::2] += grad

        x.grad += expanded.reshape(batch, 784)

    out._backward = _backward

    return out

class MNIST_Model:
    def __init__(self):
        self.l1 = Linear(784, 512)
        self.l2 = Linear(512, 256)
        self.l3 = Linear(256, 128)
        self.l4 = Linear(128, 10)

    def forward(self, x):

        h1 = self.l1(x).relu()
        h2 = self.l2(h1).relu()
        h3 = self.l3(h2).relu()
        logits = self.l4(h3)
        # Note: We don't necessarily need a Softmax Matrix node 
        # because our CrossEntropyLoss shortcut handles the math!
        return logits

    # def parameters(self):
    #     return self.l1.parameters() + self.l2.parameters()

    def parameters(self):
        return (
            self.l1.parameters()
            + self.l2.parameters()
            + self.l3.parameters()
            + self.l4.parameters()
        )


class Trainer:
    def __init__(self, model, optimizer, criterion, train_data, eval_data, batch_size) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_data = train_data
        self.eval_data = eval_data
        self.batch_size = batch_size
    def train(self, epochs):
        num_samples = self.train_data.num_samples
        for epoch in range(epochs):
            total_loss = 0
            total_correct = 0

            for i in range(0, num_samples, self.batch_size):
                self.optimizer.zero_grad()

                batch_images, batch_labels, one_hot = self.train_data.preprocess(i, self.batch_size)

                X = Matrix(batch_images)
                Y = Matrix(one_hot)

                # Forward
                logits = self.model.forward(X)


                # Loss & Accuracy
                loss, probs = self.criterion(logits, Y)
                total_loss += loss.matrix

                # Accuracy Tracking
                preds = np.argmax(probs.matrix, axis=1)

                total_correct += np.sum(preds == batch_labels)
                # Backward & Step
                loss.backward()
                self.optimizer.step()

            avg_loss = total_loss / (num_samples / self.batch_size)
            avg_acc = (total_correct / num_samples) * 100
            print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Accuracy: {avg_acc:.2f}%")

    def evaluate(self):
        num_samples = self.eval_data.num_samples
        total_loss = 0
        total_correct = 0
        total_wrong = 0
        for i in range(0, num_samples, self.batch_size):

            batch_images, batch_labels, one_hot = self.eval_data.preprocess(i, self.batch_size)

            X = Matrix(batch_images)
            Y = Matrix(one_hot)

            logits = self.model.forward(X)

            loss, probs = self.criterion(logits, Y)
            total_loss += loss.matrix

            preds = np.argmax(probs.matrix, axis=1)
            total_correct += np.sum(preds == batch_labels)
            total_wrong += np.sum(preds != batch_labels)

        avg_loss = total_loss / (num_samples / self.batch_size)
        avg_acc = (total_correct / num_samples) * 100
        print(f"Evaluation | Loss: {avg_loss:.4f} | Accuracy: {avg_acc:.2f}%")
        print(total_correct, total_wrong, num_samples)

def save_model(model, filename="model.json"):
    weights = {
        "l1_w": model.l1.w.matrix.tolist(),
        "l1_b": model.l1.b.matrix.tolist(),

        "l2_w": model.l2.w.matrix.tolist(),
        "l2_b": model.l2.b.matrix.tolist(),

        "l3_w": model.l3.w.matrix.tolist(),
        "l3_b": model.l3.b.matrix.tolist(),

        "l4_w": model.l4.w.matrix.tolist(),
        "l4_b": model.l4.b.matrix.tolist(),
    }

    with open(filename, "w") as f:
        json.dump(weights, f)


model = MNIST_Model()
optimizer = Adam(model.parameters(), lr=0.001) # Higher LR often helps MNIST early on
criterion = CrossEntropyLoss()
train_data = Dataloader('train_images.mat', 'train_labels.mat')
eval_data = Dataloader('test_images.mat', 'test_labels.mat')

batch_size = 256
epochs = 10


def main():
    trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_data=train_data,
            eval_data=eval_data,
            batch_size=batch_size
            )

    trainer.train(epochs)
    trainer.evaluate()
    save_model(model)

if __name__=='__main__':
    main()
    
    
    
def predict(index):
    optimizer.zero_grad()

    # Data Prep
    batch_imgs = train_data.images[index].reshape(-1, 784) 
    batch_labels_raw = train_data.labels[index].astype(int)
    # One-hot
    one_hot = np.zeros(10)
    one_hot[batch_labels_raw] = 1

    X = Matrix(batch_imgs)
    Y = Matrix(one_hot)

    # Forward
    logits = model.forward(X)

    _, probs = criterion(logits, Y)

    prediction = np.argmax(probs.matrix, axis=1)

    # if prediction[0] != batch_labels_raw:
        # print(f"++++++++++++++++++{index}++++++++++++++++++")
    print(prediction)

    train_data.draw(index)

# for i in range(5):
#     predict(i)




# a = Matrix([[3.1, 1.05]])
# w = Matrix([[0.6], [0.7]])
# b = Matrix([[0.9]])

# x = a * w # (1, 1)
# y = x + b # (1, 1)
# z = y ** 2 # (1, 1)

# z.backward()

# print(f"a: {a}")
# print(f"w: {w}")
# print(f"b: {b}")
# print(f"z: {z}")
