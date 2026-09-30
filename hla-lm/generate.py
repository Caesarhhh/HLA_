"""Autoregressive GDN+HLA inference from a trusted training checkpoint."""

import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer, LlamaTokenizer
from hla_config import build_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--tokenizer", required=True, help="Tokenizer used to pack the training data"
    )
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-new-tokens", type=int, default=128)
    a = p.parse_args()
    if a.max_new_tokens < 1:
        p.error("--max-new-tokens must be positive")
    model = build_model()
    state = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    state = state.get("model", state.get("state_dict", state))
    for prefix in ("_forward_module.", "module."):
        state = {k.removeprefix(prefix): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model = model.to(device="cuda", dtype=torch.bfloat16).eval()
    tokenizer_dir = Path(a.tokenizer)
    if (tokenizer_dir / "tokenizer.model").is_file() and not (
        tokenizer_dir / "tokenizer_config.json"
    ).is_file():
        tokenizer = LlamaTokenizer(vocab_file=str(tokenizer_dir / "tokenizer.model"))
    else:
        tokenizer = AutoTokenizer.from_pretrained(a.tokenizer)
    tokens = tokenizer(a.prompt, return_tensors="pt").input_ids.cuda()
    length = tokens.shape[1]
    total = length + a.max_new_tokens
    with torch.inference_mode():
        logits = model(
            tokens, max_seq_length=total, input_pos=torch.arange(length, device="cuda")
        )
        generated = []
        for i in range(a.max_new_tokens):
            token = logits[:, -1].argmax(-1, keepdim=True)
            generated.append(token.item())
            if token.item() == tokenizer.eos_token_id:
                break
            logits = model(
                token,
                max_seq_length=total,
                input_pos=torch.tensor([length + i], device="cuda"),
            )
    print(tokenizer.decode(generated, skip_special_tokens=True))


if __name__ == "__main__":
    main()
