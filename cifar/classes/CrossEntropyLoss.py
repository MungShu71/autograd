
import numpy as np

from classes.autograd import Matrix


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