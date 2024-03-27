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
        self.Mq = nn.Linear(self.input_size, self.heads * self.k_size)
        self.Mk = nn.Linear(self.input_size, self.heads * self.k_size)
        self.Mv = nn.Linear(self.input_size, self.heads * self.v_size)
        self.Wo = nn.Linear(self.heads * self.v_size, self.heads * self.v_size)

    def forward(self, x: Tensor):
        q, k, v = self._get_qkv(x)
        B, _, C, _ = v.shape
        att_weights = self._get_att_weights(
            q.view(B * self.heads, C, -1), k.view(B * self.heads, C, -1)
        )
        return self.Wo.forward(
            (att_weights.view(B, C, self.heads, -1) @ v).view(B, C, -1)
        )

    def _get_att_weights(self, q: Tensor, k: Tensor) -> Tensor:
        product = q @ k.permute(0, 2, 1)
        return nn.functional.softmax(product / sqrt(Tensor([k.shape[2]])), dim=1)

    def _get_qkv(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        B, C, _ = x.shape
        q = self.Mq.forward(x).view(B, self.heads, C, -1)
        assert isinstance(q, Tensor)
        k = self.Mk.forward(x).view(B, self.heads, C, -1)
        assert isinstance(k, Tensor)
        v = self.Mv.forward(x).view(B, self.heads, C, -1)
        assert isinstance(v, Tensor)
        return q, k, v


class MaskedMultiHeadAttentionBlock(MultiHeadAttentionBlock):
    def forward(self, x: Tensor, mask: Tensor | None = None):
        q, k, v = self._get_qkv(x)
        B, _, C, _ = v.shape
        att_weights = self._get_att_weights(
            q.view(B * self.heads, C, -1), k.view(B * self.heads, C, -1), mask
        )
        return self.Wo.forward(
            (att_weights.view(B, C, self.heads, -1) @ v).view(B, C, -1)
        )

    def _get_att_weights(
        self, q: Tensor, k: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        product = q @ k.permute(0, 2, 1)
        if mask is not None:
            B, C = mask.shape
            mask = mask.repeat(1, self.heads).view(B * self.heads, C)
            product = product + mask.unsqueeze(-1)
        return nn.functional.softmax(product / sqrt(Tensor([k.shape[2]])), dim=1)


if __name__ == "__main__":
    block = MaskedMultiHeadAttentionBlock()
    input = zeros((4, 8, 1024))
    print(block.forward(input).shape)
