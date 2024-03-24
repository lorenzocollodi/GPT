from torch import Tensor, stack, tensor
from transformers import PreTrainedTokenizerFast


class TokensDataset:
    def __init__(
        self, context_length: int, tokenizer_path: str, data_path: str
    ) -> None:
        self._text = open(data_path, "r").read()
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
        self._all_tokens = self.tokenizer.encode(self._text)
        self._n_tokens = len(self._all_tokens)
        self._context_length = context_length

    def __len__(self) -> int:
        return self._n_tokens

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        x = self._all_tokens[idx : idx + self._context_length]
        y = self._all_tokens[idx + 1]
        return tensor(x), tensor(y)

    def get_batch(self, idxs: list[int]) -> tuple[Tensor, Tensor]:
        inputs = []
        outputs = []
        for idx in idxs:
            inpt, outpt = self[idx]
            inputs.append(inpt)
            outputs.append(outpt)
        return stack(inputs), stack(outputs)
