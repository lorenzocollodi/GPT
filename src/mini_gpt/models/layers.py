from torch import Tensor, nn, sqrt, tensor, zeros


class MultiHeadAttentionBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        k_size: int,
        heads: int,
        output_size: int,
        context_length: int,
    ):
        super().__init__()
        self._input_size = input_size
        self._v_size = output_size // heads
        self._k_size = k_size
        self._context_length = context_length
        self._heads = heads
        self._output_size = output_size
        self.Mq = nn.Linear(self._input_size, self._heads * self._k_size)
        self.Mk = nn.Linear(self._input_size, self._heads * self._k_size)
        self.Mv = nn.Linear(self._input_size, self._heads * self._v_size)
        self.Wo = nn.Linear(self._heads * self._v_size, self._heads * self._v_size)

    def forward(self, x: Tensor):
        q, k, v = self._get_qkv(x)
        B, _, _, _ = v.shape
        att_weights = self._get_att_weights(
            q.view(B * self._heads, self._context_length, -1),
            k.view(B * self._heads, self._context_length, -1),
        )
        return self.Wo.forward(
            (att_weights.view(B, self._context_length, self._heads, -1) @ v).view(
                B, self._context_length, -1
            )
        )

    def _get_att_weights(self, q: Tensor, k: Tensor) -> Tensor:
        product = q @ k.permute(0, 2, 1)
        return nn.functional.softmax(
            product / sqrt(tensor([self._k_size], device=product.device)), dim=1
        )

    def _get_qkv(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        B, _, _ = x.shape
        q = self.Mq.forward(x).view(B, self._heads, self._context_length, -1)
        assert isinstance(q, Tensor)
        k = self.Mk.forward(x).view(B, self._heads, self._context_length, -1)
        assert isinstance(k, Tensor)
        v = self.Mv.forward(x).view(B, self._heads, self._context_length, -1)
        assert isinstance(v, Tensor)
        return q, k, v


class MaskedMultiHeadAttentionBlock(MultiHeadAttentionBlock):
    def forward(self, x: Tensor, mask: Tensor | None):
        q, k, v = self._get_qkv(x)
        B, _, _, _ = v.shape
        att_weights = self._get_att_weights(
            q.view(B * self._heads, self._context_length, -1),
            k.view(B * self._heads, self._context_length, -1),
            mask,
        )
        return self.Wo.forward(
            (att_weights.view(B, self._context_length, self._heads, -1) @ v).view(
                B, self._context_length, -1
            )
        )

    def _get_att_weights(
        self, q: Tensor, k: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        product = q @ k.permute(0, 2, 1)
        if mask is not None:
            B, _ = mask.shape
            mask = mask.repeat(1, self._heads).view(
                B * self._heads, self._context_length
            )
            product = product + mask.unsqueeze(-1)
        return nn.functional.softmax(
            product / sqrt(tensor([k.shape[2]], device=product.device)), dim=1
        )


if __name__ == "__main__":
    block = MaskedMultiHeadAttentionBlock()
    input = zeros((4, 8, 1024))
    print(block.forward(input).shape)
