from torch import Tensor, concat, device, stack, tensor
from transformers import PreTrainedTokenizerFast


class TokensDataset:
    def __init__(
        self,
        context_length: int,
        data_path: str,
        tokenizer: PreTrainedTokenizerFast,
        data_device: device,
    ) -> None:
        self._text = open(data_path, "r").read()
        self.tokenizer = tokenizer
        self._all_tokens = self.tokenizer.encode(self._text)
        self._n_tokens = len(self._all_tokens)
        self._context_length = context_length
        self._device = data_device

    def __len__(self) -> int:
        return self._n_tokens

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        x = self._all_tokens[idx : idx + self._context_length]
        y = self._all_tokens[idx + 1]
        return tensor(x, device=self._device), tensor([y], device=self._device)

    def get_batch(self, idxs: list[int] | Tensor) -> tuple[Tensor, Tensor]:
        if isinstance(idxs, Tensor):
            assert idxs.dim() == 1
        inputs = []
        outputs = []
        for idx in idxs:
            inpt, outpt = self[idx]
            inputs.append(inpt)
            outputs.append(outpt)
        return stack(inputs), concat(outputs, axis=0)
