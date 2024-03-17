import argparse
from pathlib import Path
from urllib.request import urlopen


def download_data(url: str, dest: Path, file_name: str) -> None:
    dest.mkdir(exist_ok=True)
    with urlopen(url) as response:
        text_data = response.read()
    with open(dest / file_name, 'wb') as write_file:
        write_file.write(text_data)

def main(args: argparse.Namespace) -> None:
    download_data(args.url, Path(args.dest), args.file_name)


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
        "--file-name",
        help="Name of the file to download",
        default="tiny_shakespeare.txt",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--dest",
        help="Destination folder",
        default="data/",
        type=str,
        required=False,
    )
    args = parser.parse_args()
    main(args)
