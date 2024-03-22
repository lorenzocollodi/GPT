import argparse
from pathlib import Path
from urllib.request import urlopen

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


def download_data(url: str, dest: Path) -> str:
    dest.mkdir(exist_ok=True)
    with urlopen(url) as response:
        text_data = response.read().decode("utf-8")
    with open(dest / "raw.txt", "w") as write_file:
        write_file.write(text_data)
    return text_data


def make_splits(text_data: str) -> tuple[str, str, str]:
    lines = text_data.split("\n")
    tot_lines = len(lines)
    train = "\n".join(lines[: int(tot_lines * 0.7)])
    val = "\n".join(lines[int(tot_lines * 0.7) : int(tot_lines * 0.9)])
    test = "\n".join(lines[int(tot_lines * 0.9) :])
    return train, val, test


def make_BPE(
    train_data: str,
    alphabet: list[str],
    save_path: str,
) -> dict[str, int]:
    print(save_path)
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.model = models.BPE()
    trainer = trainers.BpeTrainer(
        vocab_size=25000,
        initial_alphabet=alphabet,
        special_tokens=["<|endoftext|>"],
    )
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train([train_data], trainer=trainer)
    tokenizer.save(save_path)


def main(args: argparse.Namespace) -> None:
    data_path = Path(args.data_path)
    model_path = Path(args.model_path)
    text_data = download_data(args.url, data_path)
    alphabet = sorted(list(set(text_data)))
    train, val, test = make_splits(text_data)
    open(data_path / "train.txt", "w").write(train)
    open(data_path / "val.txt", "w").write(val)
    open(data_path / "test.txt", "w").write(test)
    make_BPE(
        str(data_path / "train.txt"),
        alphabet,
        str(model_path / "tokenizer.json"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="File Downlaoder",
        description="Download a file to the data/ folder."
        "By default download Tiny Shakespeare",
    )
    parser.add_argument(
        "--url",
        help="URL for the dataset to download",
        default="https://raw.githubusercontent.com/karpathy/char"
        "-rnn/master/data/tinyshakespeare/input.txt",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--data-path",
        help="Data folder",
        default="data/",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--model-path",
        help="Model folder",
        default="models/",
        type=str,
        required=False,
    )
    args = parser.parse_args()
    main(args)
