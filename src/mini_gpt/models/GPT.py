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
        q, k, v = self._get_qkv(x)
        B, _, C, _ = v.shape
        att_weights = self._get_att_weights(
            q.view(B * self.heads, C, -1), k.view(B * self.heads, C, -1)
        )
        return (att_weights.view(B, C, self.heads, -1) @ v).view(B, C, -1)

    def _get_att_weights(self, q: Tensor, k: Tensor) -> Tensor:
        product = q @ k.permute(0, 2, 1)
        return nn.functional.softmax(product / sqrt(Tensor([k.shape[2]])), dim=1)

    def _get_qkv(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        B, C, _ = x.shape
        q = (x @ self.Mq).view(B, self.heads, C, -1)
        assert isinstance(q, Tensor)
        k = (x @ self.Mk).view(B, self.heads, C, -1)
        assert isinstance(k, Tensor)
        v = (x @ self.Mv).view(B, self.heads, C, -1)
        assert isinstance(v, Tensor)
        return q, k, v


class MaskedMultiHeadAttentionBlock(MultiHeadAttentionBlock):
    def forward(self, x: Tensor):
        if not self.training:
            return super().forward(x)
        q, k, v = self._get_qkv(x)
        B, _, C, _ = v.shape
        att_weights = (
            self._get_att_weights(
                q.view(B * self.heads, C, -1), k.view(B * self.heads, C, -1)
            )
            .unsqueeze(1)
            .repeat(1, C, 1, 1, 1)
        )
        att_weights.permute(1, 0, 2, 3, 4).view(C, -1).triu().view(
            C, B, C, self.heads, -1
        ).view(B * C, self.heads, C, -1)
        v = v.repeat(C, 1, 1, 1)
        att_weights.view(B, C, self.heads, -1).repeat(C, 1, 1, 1)
        return (att_weights.view(B * C, C, self.heads, -1) @ v).view(B * C, C, -1)


class GPT(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        pass


if __name__ == "__main__":
    block = MaskedMultiHeadAttentionBlock()
    input = zeros((4, 8, 1024))
    print(block.forward(input).shape)
