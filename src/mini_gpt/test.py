import argparse

from torch import cuda, device, load, no_grad
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from mini_gpt.configs import TestConfig
from mini_gpt.datasets.batch_operators import expand_gt, get_mask, time_expand
from mini_gpt.datasets.dataset import TokensDataset
from mini_gpt.models.GPT import GPT


def parse_arguments() -> TestConfig:
    parser = argparse.ArgumentParser(
        prog="GPT tester", description="Test a GPT checkpoint"
    )
    parser.add_argument("--tokenizer", type=str, default="models/tokenizer.json")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=8)
    parser.add_argument("--checkpoint", type=str, default="models/random.pt")
    parser.add_argument("--data", type=str, default="data/test.txt")
    parser.add_argument("--log", action="store_true", default=False)
    arguments = {key: value for key, value in parser.parse_args()._get_kwargs()}
    return TestConfig(**arguments)


def test(args: TestConfig):
    if cuda.is_available():
        data_device = device("cuda")
    else:
        data_device = device("cpu")

    writer = SummaryWriter()
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer)
    test_dataset = TokensDataset(args.context_length, args.data, tokenizer, data_device)
    num_steps = len(test_dataset) // args.batch_size
    gpt_model = GPT().to(data_device)
    gpt_model.load_state_dict(
        state_dict=load(args.checkpoint, map_location=data_device)
    )
    with tqdm(
        range(0, len(test_dataset), args.batch_size), total=num_steps
    ) as progress_bar:
        for step in progress_bar:
            gpt_model.eval()
            batch_idxs = [idx for idx in range(step, step + args.batch_size)]

            batch_x, batch_y = test_dataset.get_batch(batch_idxs)
            batch_y = expand_gt(batch_x, batch_y)
            batch_x = time_expand(batch_x)
            mask = get_mask(batch_x)
            with no_grad():
                loss = gpt_model.forward(
                    batch_x.reshape(-1, args.context_length),
                    batch_y,
                    mask.reshape(-1, args.context_length),
                )
                progress_bar.set_postfix(test_loss=loss.item())
        if args.log:
            writer.add_scalar("loss/test", loss / num_steps, global_step=step)


if __name__ == "__main__":
    args = parse_arguments()
    test(args)
