from mini_gpt.datasets.dataset import TokensDataset


def train():
    dataset = TokensDataset(8, 4, "models/tokenizer.json", "data/train.txt")
    print(dataset.get_batch(3))

if __name__ == "__main__":
    train()