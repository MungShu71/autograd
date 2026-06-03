from classes.LayerNorm import LayerNorm
from classes.FeedFoward import FeedForward
from classes.MultiHeadAtt import MultiHeadAttention


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