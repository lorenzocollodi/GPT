from torch import optim, randint
from transformers import PreTrainedTokenizerFast

from mini_gpt.datasets.batch_operators import expand_gt, get_mask, time_expand
from mini_gpt.datasets.dataset import TokensDataset
from mini_gpt.models.GPT import GPT

TOKENIZER_PATH = "models/tokenizer.json"
TRAIN_DATA_PATH = "data/train.txt"
N_EPOCHS = 1000
B_SIZE = 4
CONTEXT_LENGTH = 8


def train():
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_PATH)
    dataset = TokensDataset(CONTEXT_LENGTH, TRAIN_DATA_PATH, tokenizer)
    gpt_model = GPT()
    batch_x, batch_y = dataset.get_batch([4, 100, 20])
    for param in gpt_model.parameters():
        param.requires_grad = True
    gpt_model.train()
    optimizer = optim.AdamW(gpt_model.parameters())
    for _ in range(N_EPOCHS):
        batch_idxs = randint(high=len(dataset) - CONTEXT_LENGTH, size=(B_SIZE,))

        batch_x, batch_y = dataset.get_batch(batch_idxs)
        batch_y = expand_gt(batch_x, batch_y)
        batch_x = time_expand(batch_x)
        mask = get_mask(batch_x)
        loss = gpt_model.forward(
            batch_x.reshape(-1, CONTEXT_LENGTH),
            batch_y,
            mask.reshape(-1, CONTEXT_LENGTH),
        )
        for param in gpt_model.parameters():
            param.grad = None
        loss.backward()
        optimizer.step()
        print(loss)


if __name__ == "__main__":
    train()
