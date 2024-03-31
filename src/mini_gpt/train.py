import argparse

from tqdm import tqdm
from torch import cuda, device, no_grad, optim, randint
from torch.utils.tensorboard import SummaryWriter
from transformers import PreTrainedTokenizerFast

from mini_gpt.configs import TrainConfig
from mini_gpt.datasets.batch_operators import expand_gt, get_mask, time_expand
from mini_gpt.datasets.dataset import TokensDataset
from mini_gpt.models.GPT import GPT


def parse_arguments() -> TrainConfig:
    parser = argparse.ArgumentParser(
        prog="GPT trainer", description="Train your own GPT model"
    )
    parser.add_argument("--tokenizer", type=str, default="models/tokenizer.json")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--val-every", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=8)
    parser.add_argument("--train-data", type=str, default="data/train.txt")
    parser.add_argument("--val-data", type=str, default="data/val.txt")
    parser.add_argument("--log", action="store_true", default=False)
    arguments = {key: value for key, value in parser.parse_args()._get_kwargs()}
    return TrainConfig(**arguments)


def train(args: TrainConfig):
    if cuda.is_available():
        data_device = device("cuda")
    else:
        data_device = device("cpu")

    num_steps = args.epochs*args.val_every
    writer = SummaryWriter()
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer)
    train_dataset = TokensDataset(
        args.context_length, args.train_data, tokenizer, data_device
    )
    val_dataset = TokensDataset(
        args.context_length, args.val_data, tokenizer, data_device
    )
    gpt_model = GPT().to(data_device)
    for param in gpt_model.parameters():
        param.requires_grad = True
    optimizer = optim.AdamW(gpt_model.parameters())
    with tqdm(range(num_steps), total=num_steps) as progress_bar:
        for step in progress_bar:
            gpt_model.train()
            batch_idxs = randint(
                high=len(train_dataset) - args.context_length, size=(args.batch_size,)
            )

            batch_x, batch_y = train_dataset.get_batch(batch_idxs)
            batch_y = expand_gt(batch_x, batch_y)
            batch_x = time_expand(batch_x)
            mask = get_mask(batch_x)
            loss = gpt_model.forward(
                batch_x.reshape(-1, args.context_length),
                batch_y,
                mask.reshape(-1, args.context_length),
            )
            progress_bar.set_postfix(train_loss = loss.item())
            if args.log:
                writer.add_scalar("loss/train", loss, global_step=step)
            for param in gpt_model.parameters():
                param.grad = None
            loss.backward()
            optimizer.step()
            if (step + 1) % args.val_every == 0:
                gpt_model.eval()
                for step in range(len(val_dataset)):
                    batch_x, batch_y = val_dataset[step]
                    val_loss = 0
                    with no_grad():
                        batch_y = expand_gt(batch_x.unsqueeze(0), batch_y)
                        batch_x = time_expand(batch_x.unsqueeze(0))
                        mask = get_mask(batch_x)
                        val_loss += gpt_model.forward(
                            batch_x.reshape(-1, args.context_length),
                            batch_y,
                            mask.reshape(-1, args.context_length),
                        )
                    val_loss /= len(val_dataset)
                    progress_bar.set_postfix(train_loss = val_loss)
                    if args.log:
                        writer.add_scalar("loss/val", val_loss.item(), global_step=(step+1)//args.val_every)


if __name__ == "__main__":
    args = parse_arguments()
    train(args)
