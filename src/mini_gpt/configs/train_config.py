from pydantic.dataclasses import dataclass


@dataclass(frozen=True, validate_on_init=True)
class TrainConfig:
    tokenizer: str
    epochs: int
    val_every: int
    batch_size: int
    context_length: int
    train_data: str
    val_data: str
