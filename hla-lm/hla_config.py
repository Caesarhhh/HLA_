"""Paper configuration for the GDN-based HLA model (C=256, P=16)."""

from lit_gpt import Config, GPT


def paper_config(tiny=False):
    c = Config.from_name("GatedDeltaNet_GLA_GDN_Release_1.3B")
    c.block_size = 4096
    c.mix_chunk_size = 256
    c.mix_pool_size = 16
    c.mix_apply_gdn_decay = False
    c.gla_router_route_source = "raw_qk"
    c.gla_router_use_logmean = True
    c.gla_router_use_rope = False
    c.gla_router_pool_self_attention = True
    c.gla_router_gate = "affine_sigmoid"
    c.gla_router_sigmoid_bias = 2.2
    if tiny:
        c.n_layer = 2
        c.n_embd = 256
        c.n_head = 4
        c.n_query_groups = 4
        c.gated_delta_num_heads = 2
        c.gated_delta_head_dim = 128
        c.intermediate_size = 512
        c.vocab_size = 256
        c.padded_vocab_size = 256
    return c


def build_model(tiny=False):
    model = GPT(paper_config(tiny))
    for module in model.modules():
        if hasattr(module, "router_gate"):
            module.router_current_always_on = True
            module.router_sigmoid_temperature = 1.0
    return model
