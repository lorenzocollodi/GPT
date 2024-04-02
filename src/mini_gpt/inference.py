import argparse

from torch import cuda, device, load, multinomial, no_grad
from torch.nn.functional import softmax
from transformers import PreTrainedTokenizerFast

from mini_gpt.configs import InferenceConfig
from mini_gpt.inference import QueuedText
from mini_gpt.models.GPT import GPT


def parse_arguments() -> InferenceConfig:
    parser = argparse.ArgumentParser(
        prog="GPT tester", description="Test a GPT checkpoint"
    )
    parser.add_argument("--tokenizer", type=str, default="models/tokenizer.json")
    parser.add_argument("--context-length", type=int, default=8)
    parser.add_argument("--checkpoint", type=str, default="models/random.pt")
    parser.add_argument("--prompt", type=str, default="A")
    arguments = {key: value for key, value in parser.parse_args()._get_kwargs()}
    return InferenceConfig(**arguments)


def test(args: InferenceConfig):
    if cuda.is_available():
        data_device = device("cuda")
    else:
        data_device = device("cpu")

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer)
    text = QueuedText(args.prompt, args.context_length, tokenizer, data_device)
    gpt_model = GPT().to(data_device)
    gpt_model.load_state_dict(
        state_dict=load(args.checkpoint, map_location=data_device)
    )
    predicted_character = -1
    gpt_model.eval()
    while predicted_character != 0:
        inpt, mask = text.poll()
        with no_grad():
            logits = gpt_model.forward(
                inpt,
                mask=mask,
            )
            probs = softmax(logits)
            predicted_character = multinomial(probs, 1).item()
            text.add_token(predicted_character)
            print(text.decode())


if __name__ == "__main__":
    args = parse_arguments()
    test(args)
