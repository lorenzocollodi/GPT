from torch import nn, Tensor, sqrt, tensordot, zeros


class AttentionBlock(nn.Module):
    def __init__(self, input_size: int = 1024, k_size: int = 64, heads: int = 8, output_size: int = 1024):
        self.input_size = input_size
        self.v_size = output_size / heads
        self.k_size = k_size
        self.heads = 8
        self.output_size = output_size
        self.Mq = zeros((self.input_size, self.heads, self.k_size))
        self.Mk = zeros((self.input_size, self.heads, self.k_size))
        self.Mv = zeros((self.input_size, self.heads, self.v_size))        
    
    def forward(self, x: Tensor):
        Q = x @ self.Mq
        K = x @ self.Mk
        V = x @ self.Mv
        head_weights = nn.functional.softmax(tensordot(Q, K.T) / sqrt(self.k_size), dim=1)
        return head_weights @ V.T

class GPT(nn.Model):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        pass