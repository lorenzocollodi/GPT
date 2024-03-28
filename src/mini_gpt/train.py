from torch import cuda, device, no_grad, optim, randint
from torch.utils.tensorboard import SummaryWriter
from transformers import PreTrainedTokenizerFast

from mini_gpt.datasets.batch_operators import expand_gt, get_mask, time_expand
from mini_gpt.datasets.dataset import TokensDataset
from mini_gpt.models.GPT import GPT

TOKENIZER_PATH = "models/tokenizer.json"
TRAIN_DATA_PATH = "data/train.txt"
VAL_DATA_PATH = "data/val.txt"
N_EPOCHS = 100
N_STEPS = 1000
B_SIZE = 4
CONTEXT_LENGTH = 8


def train():
    if cuda.is_available():
        data_device = device("cuda")
    else:
        data_device = device("cpu")

    writer = SummaryWriter()
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_PATH)
    train_dataset = TokensDataset(
        CONTEXT_LENGTH, TRAIN_DATA_PATH, tokenizer, data_device
    )
    val_dataset = TokensDataset(CONTEXT_LENGTH, VAL_DATA_PATH, tokenizer, data_device)
    gpt_model = GPT().to(data_device)
    for param in gpt_model.parameters():
        param.requires_grad = True
    optimizer = optim.AdamW(gpt_model.parameters())
    for epoch in range(N_EPOCHS):
        gpt_model.train()
        for step in range(N_STEPS):
            step = epoch * N_STEPS + step
            batch_idxs = randint(
                high=len(train_dataset) - CONTEXT_LENGTH, size=(B_SIZE,)
            )

            batch_x, batch_y = train_dataset.get_batch(batch_idxs)
            batch_y = expand_gt(batch_x, batch_y)
            batch_x = time_expand(batch_x)
            mask = get_mask(batch_x)
            loss = gpt_model.forward(
                batch_x.reshape(-1, CONTEXT_LENGTH),
                batch_y,
                mask.reshape(-1, CONTEXT_LENGTH),
            )
            writer.add_scalar("loss/train", loss, global_step=step)
            for param in gpt_model.parameters():
                param.grad = None
            loss.backward()
            print(loss)
            optimizer.step()
        gpt_model.eval()
        for step in range(len(val_dataset)):
            batch_x, batch_y = val_dataset[step]
            val_loss = 0
            with no_grad():
                batch_y = expand_gt(batch_x.unsqueeze(0), batch_y)
                batch_x = time_expand(batch_x.unsqueeze(0))
                mask = get_mask(batch_x)
                val_loss += gpt_model.forward(
                    batch_x.reshape(-1, CONTEXT_LENGTH),
                    batch_y,
                    mask.reshape(-1, CONTEXT_LENGTH),
                )
            val_loss /= len(val_dataset)
            writer.add_scalar("loss/val", val_loss, global_step=epoch)


if __name__ == "__main__":
    train()
