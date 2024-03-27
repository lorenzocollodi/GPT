from torch import Tensor, concat, long, nn, zeros

from mini_gpt.models.blocks import MaskedMultiHeadAttentionBlock

nn.MultiheadAttention


class GPT(nn.Module):
    def __init__(
        self,
        depth: int = 6,
        num_tokens: int = 27000,
        positional_encoding: int = 64,
        encoding_dimension: int = 960,
        context_length: int = 8,
    ):
        super().__init__()
        self._num_tokens = num_tokens
        self.seq_blocks = nn.Sequential(
            *[
                MaskedMultiHeadAttentionBlock(
                    input_size=positional_encoding + encoding_dimension
                )
                for _ in range(depth)
            ]
        )
        self.meaning_embeddings = nn.Embedding(num_tokens, encoding_dimension)
        self.pos_embeddings = nn.Embedding(context_length, positional_encoding)
        self.out_mlp = nn.Linear(
            context_length * (positional_encoding + encoding_dimension), num_tokens
        )
        self._CELoss = nn.CrossEntropyLoss()

    def forward(
        self, x: Tensor, gt: Tensor | None = None, mask: Tensor | None = None
    ) -> Tensor:
        B, C = x.shape
        embeddings = self.meaning_embeddings.forward(x)
        pos_embeddings = self.pos_embeddings.forward(
            Tensor([pos_idx for pos_idx in range(C)]).long()
        )
        assert isinstance(embeddings, Tensor)
        assert isinstance(pos_embeddings, Tensor)
        inpt = concat((embeddings, pos_embeddings.repeat(B, 1, 1)), 2)

        for block in self.seq_blocks:
            inpt = block.forward(inpt, mask)
        assert isinstance(inpt, Tensor)
        output = inpt.view(B, -1)
        logits = self.out_mlp(output)
        assert isinstance(logits, Tensor)
        if gt is None:
            return logits
        loss = self._CELoss(logits, gt)
        return loss


if __name__ == "__main__":
    gpt = GPT()
    zeros((3, 8), dtype=long)
    print(gpt.forward(zeros((3, 8), dtype=long)).shape)
