from torch import concat, long, zeros, Tensor, nn
from mini_gpt.models.blocks import MaskedMultiHeadAttentionBlock, MultiHeadAttentionBlock


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
        self.input_block = MaskedMultiHeadAttentionBlock(input_size = positional_encoding+encoding_dimension)
        self.seq_blocks = nn.Sequential(*[MultiHeadAttentionBlock(input_size = positional_encoding+encoding_dimension) for _ in range(depth)])
        self.meaning_embeddings = nn.Parameter(zeros(num_tokens, encoding_dimension))
        self.pos_embeddings = nn.Parameter(zeros(context_length, positional_encoding))
        self.out_mlp = nn.Linear(positional_encoding+encoding_dimension, 27000)

    def forward(self, x: Tensor) -> Tensor:
        B, _ = x.shape
        embeddings = self.meaning_embeddings[x]
        embeddings = concat((embeddings, self.pos_embeddings.repeat(B, 1, 1)), 2)
        output = self.input_block.forward(embeddings)
        output = self.seq_blocks(output)
        logits = self.out_mlp(output)
        return logits


if __name__ == "__main__":
    gpt = GPT()
    zeros((3, 8), dtype = long)
    print(gpt.forward(zeros((3, 8), dtype = long)).shape)
