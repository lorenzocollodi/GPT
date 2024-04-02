from torch import Tensor, concat, device, long, tensor, zeros
from transformers import PreTrainedTokenizerFast


class QueuedText:
    def __init__(
        self,
        prompt: str,
        context_length: int,
        tokenizer: PreTrainedTokenizerFast,
        data_device: device,
    ) -> None:
        self._tokenized_text = tokenizer.encode(prompt)
        self._tokenizer = tokenizer
        self._context_length = context_length
        self._data_device = data_device

    def poll(self) -> tuple[Tensor, Tensor | None]:
        """Return latest window plus the mask for masked attention if needed."""
        if len(self._tokenized_text) < 8:
            mask = zeros((1, 8))
            mask[:, : len(self._tokenized_text)] = 1
            data = tensor([self._tokenized_text[-self._context_length :]], dtype=long)
            data_padding = zeros((1, 8 - len(self._tokenized_text)), dtype=long)
            return concat((data, data_padding), dim=1), mask
        else:
            return (
                tensor([self._tokenized_text[-self._context_length :]], dtype=long),
                None,
            )

    def add_token(self, token: int) -> None:
        self._tokenized_text.append(token)

    def decode(self) -> str:
        return self._tokenizer.decode(self._tokenized_text)
