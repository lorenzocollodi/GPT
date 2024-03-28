from torch import Tensor, nn

from mini_gpt.models.layers import MaskedMultiHeadAttentionBlock


class GPTBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        k_size: int,
        heads: int,
        output_size: int,
        context_length: int,
    ) -> None:
        super().__init__()
        self._input_size = input_size
        self._k_size = k_size
        self._heads = heads
        self._context_length = context_length
        self._output_size = output_size
        self.mh_att = MaskedMultiHeadAttentionBlock(
            input_size=input_size,
            k_size=k_size,
            heads=heads,
            output_size=output_size,
            context_length=context_length,
        )
        self.out_blocks = nn.Sequential(
            nn.LayerNorm(output_size),
            nn.Linear(output_size, output_size),
            nn.ReLU(),
            nn.LayerNorm(output_size),
        )

    def forward(self, inpt: Tensor, mask: Tensor | None) -> Tensor:
        inpt = self.mh_att.forward(inpt, mask)
        return self.out_blocks.forward(inpt)
