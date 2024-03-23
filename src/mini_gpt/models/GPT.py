from torch import nn, Tensor, sqrt, tensordot, zeros


class AttentionBlock(nn.Module):
    def __init__(self, input_size: int = 1024, k_size: int = 64, heads: int = 8, output_size: int = 1024):
        self.input_size = input_size
        self.v_size = output_size // heads
        self.k_size = k_size
        self.heads = heads
        self.output_size = output_size
        self.Mq = zeros((self.heads, 1, self.input_size, self.k_size))
        self.Mk = zeros((self.heads, 1, self.input_size, self.k_size))
        self.Mv = zeros((self.heads, 1, self.input_size, self.v_size))    
        self._parameters = {"Mq": self.Mq, "Mk": self.Mk, "Mv": self.Mv}
        self._modules = {}

    def forward(self, x: Tensor):
        q = x @ self.Mq
        print(q.shape)
        k = x @ self.Mk
        print(k.shape)
        v = x @ self.Mv
        print(v.shape)
        head_weights = nn.functional.softmax(tensordot(q, k.T) / sqrt(self.k_size), dim=1)
        return head_weights @ v.T


class GPT(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        pass

if __name__ == "__main__":
    block = AttentionBlock()
    input = zeros((4, 8, 1024))
    block.forward(input)
    print([param for param in block.parameters()])