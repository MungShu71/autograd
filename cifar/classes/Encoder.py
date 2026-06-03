from classes.Block import Block


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