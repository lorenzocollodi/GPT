from pydantic.dataclasses import dataclass


@dataclass(frozen=True, validate_on_init=True)
class InferenceConfig:
    tokenizer: str
    context_length: int
    prompt: str
    checkpoint: str
