from mini_gpt.datasets.batch_operators import expand_gt
from mini_gpt.datasets.dataset import TokensDataset
from mini_gpt.models.GPT import MaskedMultiHeadAttentionBlock
from torch import zeros


def train():
    dataset = TokensDataset(8, "models/tokenizer.json", "data/train.txt")
    batch_x, batch_y = dataset.get_batch([4, 100, 20])
    print(batch_x.shape)
    print(batch_y.shape)
    
    block = MaskedMultiHeadAttentionBlock()
    print(block.forward(zeros((3, 8, 1024))).shape)

    batch_y = expand_gt(batch_x, batch_y)
    print(batch_y.shape)


if __name__ == "__main__":
    train()
