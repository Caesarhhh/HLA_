# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Copyright Lightning AI. Licensed under the Apache License 2.0,
# see LICENSE file at https://github.com/Lightning-AI/litgpt/blob/main/LICENSE

from lit_gpt.model import GPT
from lit_gpt.config import Config
from lit_gpt.tokenizer import Tokenizer
from lit_gpt.fused_cross_entropy import FusedCrossEntropyLoss
try:
    from lightning_utilities.core.imports import RequirementCache
except ImportError:
    class RequirementCache:
        def __init__(self, requirement: str):
            self.requirement = requirement

        def __bool__(self) -> bool:
            return True

if not bool(RequirementCache("torch>=2.1.0dev")):
    raise ImportError(
        "Lit-GPT requires torch nightly (future torch 2.1). Please follow the installation instructions in the"
        " repository README.md"
    )
_LIGHTNING_AVAILABLE = RequirementCache("lightning>=2.1.0.dev0")


__all__ = ["GPT", "Config", "Tokenizer"]
