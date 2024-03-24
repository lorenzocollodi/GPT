from torch import randint, nn, optim

from mini_gpt.datasets.batch_operators import expand_gt
from mini_gpt.datasets.dataset import TokensDataset
from mini_gpt.models.GPT import GPT
from transformers import PreTrainedTokenizerFast


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
        batch_idxs = randint(high = len(dataset) - CONTEXT_LENGTH, size=(B_SIZE,))
        
        batch_x, batch_y = dataset.get_batch(batch_idxs)
        batch_y = expand_gt(batch_x, batch_y)
        loss = gpt_model.forward(batch_x, batch_y)
        for param in gpt_model.parameters():
            param.grad = None
        loss.backward()
        optimizer.step()    
        print(loss)



if __name__ == "__main__":
    train()
