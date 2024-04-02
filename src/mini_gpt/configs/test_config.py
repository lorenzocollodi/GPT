from pydantic.dataclasses import dataclass


@dataclass(frozen=True, validate_on_init=True)
class TestConfig:
    tokenizer: str
    batch_size: int
    context_length: int
    data: str
    log: bool
    checkpoint: str
