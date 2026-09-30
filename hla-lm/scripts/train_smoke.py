"""Run the real trainer for one optimizer step on a tiny model and synthetic packed data.

This is an execution check, not a quality benchmark or a paper training run.
"""

import copy
import os
from pathlib import Path
import runpy
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault("WANDB_MODE", "disabled")
import numpy as np
from hla_config import paper_config
from lit_gpt import Config
from lit_gpt.packed_dataset import PackedDatasetBuilder

config = paper_config(tiny=True)
Config.from_name = classmethod(lambda cls, name, **kwargs: copy.deepcopy(config))
(ROOT / "outputs").mkdir(exist_ok=True)
output = Path(tempfile.mkdtemp(prefix="train_smoke_", dir=ROOT / "outputs"))
data = output / "data"
data.mkdir(parents=True, exist_ok=True)
if len(list(data.glob("train_slim*.bin"))) < 8:
    builder = PackedDatasetBuilder(
        str(data), "train_slim_smoke", 4097 * 2, 0, vocab_size=256
    )
    builder.add_array(
        np.random.default_rng(3407).integers(0, 256, size=4097 * 2 * 8, dtype=np.uint16)
    )
    builder.write_reminder()
sys.argv = [
    str(ROOT / "pretrain.py"),
    "--train_data_dir",
    str(data),
    "--output_root",
    str(output),
    "--model_name",
    "GatedDeltaNet_GLA_GDN_Release_1.3B",
    "--train_config",
    "512x4k_100B",
    "--exp_name",
    "tiny_execution_check",
    "--micro_batch_size",
    "1",
    "--gradient_accumulation_steps_override",
    "1",
    "--max_tokens_override",
    "4096",
    "--warmup_tokens_override",
    "4096",
    "--train_num_workers",
    "0",
    "--stop_after_step",
    "1",
    "--interactive_job",
    "--save_step_checkpoints",
    "--save_step_interval",
    "1",
    "--log_step_interval",
    "1",
]
runpy.run_path(str(ROOT / "pretrain.py"), run_name="__main__")
print(
    "PASS real trainer: synthetic 4K batch, tiny model, router losses, optimizer and checkpoint"
)
