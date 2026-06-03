from classes.Linear import Linear


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