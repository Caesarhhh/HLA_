"""GPU integration check: training gradients, optimizer step, cached generation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from hla_config import build_model

torch.manual_seed(3407)
model = build_model(tiny=True).cuda().to(torch.bfloat16)
x = torch.randint(0, 256, (1, 520), device="cuda")
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
logits = model(x)
loss = torch.nn.functional.cross_entropy(
    logits[:, :-1].float().flatten(0, 1), x[:, 1:].flatten()
)
assert torch.isfinite(loss)
loss.backward()
assert all(
    torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
)
opt.step()
opt.zero_grad(set_to_none=True)
model.eval()
with torch.inference_mode():
    ref = model(x)
    model.kv_caches = []
    cached = model(x, max_seq_length=522, input_pos=torch.arange(520, device="cuda"))
    torch.testing.assert_close(cached[:, -1], ref[:, -1], atol=0.12, rtol=0.12)
    step = model(
        cached[:, -1].argmax(-1, keepdim=True),
        max_seq_length=522,
        input_pos=torch.tensor([520], device="cuda"),
    )
    assert torch.isfinite(step).all()
torch.cuda.synchronize()
print(
    f"PASS training + cached prefill/decode; loss={loss.item():.5f}; GPU={torch.cuda.get_device_name()}"
)
