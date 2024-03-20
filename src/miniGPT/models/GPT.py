from torch import nn, Tensor


class GPT(nn.Model):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        pass