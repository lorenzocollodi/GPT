from torch import stack, tensor, Tensor
from transformers import PreTrainedTokenizerFast


class TokensDataset():
    def __init__(self, context_length: int, batch_size: int, tokenizer_path: str, data_path: str) -> None:
        self._text = open(data_path, 'r').read()
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
        self._all_tokens = self.tokenizer.encode(self._text)
        self._n_tokens = len(self._all_tokens)
        self._context_length = context_length
        self._batch_size = batch_size

    
    def __len__(self) -> int:
        return self._n_tokens


    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        x = self._all_tokens[idx:idx+self._context_length]
        y = self._all_tokens[idx+1:idx+self._context_length+1]
        return tensor(x), tensor(y)


    def get_batch(self, idxs: list[int] | int) -> tuple[Tensor, Tensor]:
        if isinstance(idxs, int):
            assert idxs < len(self) - self._context_length - self._batch_size
            idxs = [idx for idx in range(idxs, idxs+self._batch_size)]
        assert len(idxs) == self._batch_size
        inputs = []
        outputs = []
        for idx in idxs:
            inpt, outpt = self[idx]
            inputs.append(inpt)
            outputs.append(outpt)
        return stack(inputs), stack(outputs)