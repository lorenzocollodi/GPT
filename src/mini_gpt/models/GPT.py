from torch import Tensor, nn, sqrt, zeros


class MultiHeadAttentionBlock(nn.Module):
    def __init__(
        self,
        input_size: int = 1024,
        k_size: int = 64,
        heads: int = 8,
        output_size: int = 1024,
    ):
        super().__init__()
        self.input_size = input_size
        self.v_size = output_size // heads
        self.k_size = k_size
        self.heads = heads
        self.output_size = output_size
        self.Mq = nn.Parameter(zeros(self.input_size, self.heads * self.k_size))
        self.Mk = nn.Parameter(zeros(self.input_size, self.heads * self.k_size))
        self.Mv = nn.Parameter(zeros(self.input_size, self.heads * self.v_size))

    def forward(self, x: Tensor):
        B, C, _ = x.shape
        q = (x @ self.Mq).view(B, self.heads, C, -1)
        assert isinstance(q, Tensor)
        k = (x @ self.Mk).view(B, self.heads, C, -1)
        assert isinstance(k, Tensor)
        v = (x @ self.Mv).view(B, self.heads, C, -1)
        assert isinstance(v, Tensor)
        product = q.view(B * self.heads, C, -1) @ k.permute(0, 1, 3, 2).view(
            B * self.heads, -1, C
        )
        normalised_product = product / sqrt(Tensor([self.k_size]))
        head_weights = nn.functional.softmax(normalised_product, dim=1)
        return (head_weights.view(B, C, self.heads, -1) @ v).view(B, C, -1)


class GPT(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        pass


if __name__ == "__main__":
    block = MultiHeadAttentionBlock()
    input = zeros((4, 8, 1024))
    print(block.forward(input).shape)
