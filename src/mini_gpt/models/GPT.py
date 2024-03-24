from torch import Tensor, nn


class GPT(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        pass


if __name__ == "__main__":
    pass
