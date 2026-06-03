import time
import json
import numpy as np

from classes.CrossEntropyLoss import CrossEntropyLoss
from classes.Embedder import Embedder
from classes.Encoder import Encoder
from classes.LayerNorm import LayerNorm
from classes.Linear import Linear
from classes.MultiHeadAtt import MultiHeadAttention
from classes.SGD import SGD
from classes.autograd import Matrix
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

        
        self.num_samples, self.in_channel, self.h, self.w = self.images.shape
    
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
                print(f"\x1b[38;2;{r1};{g1};{b1}m\x1b[48;2;{r2};{g2};{b2}m▀", end="")
            print("\x1b[0m")
            
        print(f"Label: {classes[self.labels[index]]}")

    """

    def draw(self, index=0):
        # image = self.images[index]
        image = self.images.reshape(-1, self.in_channel * self.h * self.w)[index]
        for y in range(self.h):
            for x in range(self.w):
                num = image[x+y*self.w]
                col = 232 + int(num * 23)
                print(f"\x1b[48;5;{col}m  ", end="")
            print("\x1b[0m")
        print("\x1b[0m")
        print("Label: ", end='')
        for i in range(10):
            print(1 if i==self.labels[index] else 0, end=' ')
        print()
        
    """
        
    def preprocess(self, start, batch_size):
        x = self.images[start : start + batch_size] / 255.0
        y = self.labels[start : start + batch_size].astype(np.int32)

        one_hot = np.zeros((len(y), 10), dtype=np.float32)
        one_hot[np.arange(len(y)), y] = 1.0

        return x, y, one_hot



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

def save_model(model, filename="transformer_model.json"):
    params = {}

    for i, p in enumerate(model.parameters()):
        params[f"param_{i}"] = p.matrix.tolist()

    with open(filename, "w") as f:
        json.dump(params, f)


def main():
    #train_data = Dataloader("mnist_train_images.npy", "mnist_train_labels.npy")

    #eval_data = Dataloader("mnist_test_images.npy", "mnist_test_labels.npy")
   
    train_data  = Dataloader("cifar10/cifar10_train_images.npy", "cifar10/cifar10_train_labels.npy")
    eval_data  = Dataloader("cifar10/cifar10_test_images.npy", "cifar10/cifar10_train_labels.npy")
    train_data.draw(3)
    vit = VIT(
        image_size=32, 
        patch_size=4,
        in_channels=3, 
        hidden_size=48,
        intermediate_size=48 * 4,
        num_heads=4,
        depth=4,
        num_classes=10,
    )

    optimizer = SGD(vit.parameters(), lr=3e-4)

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
    #save_model(vit)
    #trainer.evaluate()


if __name__ == "__main__":
    np.random.seed(67)
    #main()
    
    
    # Flash Attention Testing #
    
    '''
    q_base = Matrix(np.random.randn(32, 65, 48).astype(np.float32))
    k_base = Matrix(np.random.randn(32, 65, 48).astype(np.float32))
    v_base = Matrix(np.random.randn(32, 65, 48).astype(np.float32))
    
   
    q_mha = q_base.reshape(32, 65, 4, 12).transpose(0, 2, 1, 3)
    k_mha = k_base.reshape(32, 65, 4, 12).transpose(0, 2, 1, 3)
    v_mha = v_base.reshape(32, 65, 4, 12).transpose(0, 2, 1, 3)
    
    kt_mha = k_mha.transpose(0, 1, 3, 2)                         # (32, 4, 12, 65)
    q_kt_mha = q_mha.matmul(kt_mha) * (1.0 / np.sqrt(12))        # (32, 4, 65, 65)
    softmax_mha = q_kt_mha.softmax()                             # (32, 4, 65, 65)
    output_mha = softmax_mha.matmul(v_mha)
    out = output_mha.transpose(0, 2, 1, 3).reshape(32, 65, 48)
    loss = out * 2 
    loss.backward()
  
    
    q_mha1 = q_base.reshape(32, 65, 4, 12).transpose(0, 2, 1, 3)
    k_mha1 = k_base.reshape(32, 65, 4, 12).transpose(0, 2, 1, 3)
    v_mha1 = v_base.reshape(32, 65, 4, 12).transpose(0, 2, 1, 3)
    
    flash = MultiHeadAttention.flash_attention(q_mha1, k_mha1, v_mha1, block_size=16)
    loss = flash * 2
    loss.backward()
    
    print("attention output")
    print("match:", np.allclose(out.matrix, flash.matrix, atol=1e-5))
    print("max diff:", np.abs(out.matrix - flash.matrix).max())
    
    print("Q matrixs")
    print("match:", np.allclose(q_mha.matrix, q_mha1.matrix, atol=1e-5))
    print("max diff:", np.abs(q_mha.matrix, q_mha1.matrix,).max())
    
    print("K matrixs")
    print("match:", np.allclose(k_mha.matrix, k_mha1.matrix, atol=1e-5))
    print("max diff:", np.abs(k_mha.matrix, k_mha1.matrix,).max())

    print("V matrixs")
    print("match:", np.allclose(v_mha.matrix, v_mha1.matrix, atol=1e-5))
    print("max diff:", np.abs(v_mha.matrix, v_mha1.matrix,).max())
    
    print("attention output")
    print("match:", np.allclose(out.grad, flash.grad, atol=1e-5))
    print("max diff:", np.abs(out.grad - flash.grad).max())
    
    print("Q grads")
    print("match:", np.allclose(q_mha.grad, q_mha1.grad, atol=1e-5))
    print("max diff:", np.abs(q_mha.grad, q_mha1.grad,).max())
    
    print("K grads")
    print("match:", np.allclose(k_mha.grad, k_mha1.grad, atol=1e-5))
    print("max diff:", np.abs(k_mha.grad, k_mha1.grad,).max())

    print("V grads")
    print("match:", np.allclose(v_mha.grad, v_mha1.grad, atol=1e-5))
    print("max diff:", np.abs(v_mha.grad, v_mha1.grad,).max())
    '''
    
    
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
