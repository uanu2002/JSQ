import torch
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaDecoderLayer
from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer, OPTForCausalLM
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP
from jsq.fake_quant import W8A8Linear

@torch.no_grad()
def quantize_model(model, weight_quant='per_tensor', act_quant='per_tensor', quantize_bmm_input=True):
    for name, m in model.named_modules():
        if isinstance(m, OPTDecoderLayer):
            m.fc1 = W8A8Linear.from_float(m.fc1, weight_quant=weight_quant, act_quant=act_quant)
            m.fc2 = W8A8Linear.from_float(m.fc2, weight_quant=weight_quant, act_quant=act_quant)
        elif isinstance(m, OPTAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W8A8Linear.from_float(
                m.q_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.k_proj = W8A8Linear.from_float(
                m.k_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.v_proj = W8A8Linear.from_float(
                m.v_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.out_proj = W8A8Linear.from_float(m.out_proj, weight_quant=weight_quant, act_quant=act_quant)
        elif isinstance(m, LlamaAttention):
            m.q_proj = W8A8Linear.from_float(
                m.q_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.k_proj = W8A8Linear.from_float(
                m.k_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.v_proj = W8A8Linear.from_float(
                m.v_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.o_proj = W8A8Linear.from_float(m.o_proj, weight_quant=weight_quant, act_quant=act_quant)
        elif isinstance(m, LlamaMLP):
            m.gate_proj = W8A8Linear.from_float(m.gate_proj, weight_quant=weight_quant, act_quant=act_quant)
            m.down_proj = W8A8Linear.from_float(m.down_proj, weight_quant=weight_quant, act_quant=act_quant)
            m.up_proj = W8A8Linear.from_float(m.up_proj, weight_quant=weight_quant, act_quant=act_quant)
    return model


@torch.no_grad()
def quantize_layer(module, nbits, weight_quant='per_channel', act_quant='per_token', quantize_bmm_input=True):
    for name, m in module.named_modules():
        if isinstance(m, OPTDecoderLayer):
            m.fc1 = W8A8Linear.from_float(m.fc1, weight_quant=weight_quant, act_quant=act_quant)
            m.fc2 = W8A8Linear.from_float(m.fc2, weight_quant=weight_quant, act_quant=act_quant)
        elif isinstance(m, OPTAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W8A8Linear.from_float(
                m.q_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.k_proj = W8A8Linear.from_float(
                m.k_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.v_proj = W8A8Linear.from_float(
                m.v_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.out_proj = W8A8Linear.from_float(m.out_proj, weight_quant=weight_quant, act_quant=act_quant)
        elif isinstance(m, LlamaAttention):
            m.q_proj = W8A8Linear.from_float(
                m.q_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input, nbits=nbits)
            m.k_proj = W8A8Linear.from_float(
                m.k_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input, nbits=nbits)
            m.v_proj = W8A8Linear.from_float(
                m.v_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input, nbits=nbits)
            m.o_proj = W8A8Linear.from_float(m.o_proj, weight_quant=weight_quant, act_quant=act_quant, nbits=nbits)
        elif isinstance(m, LlamaDecoderLayer):
            m.mlp.gate_proj = W8A8Linear.from_float(m.mlp.gate_proj, weight_quant=weight_quant, act_quant=act_quant, nbits=nbits)
            m.mlp.down_proj = W8A8Linear.from_float(m.mlp.down_proj, weight_quant=weight_quant, act_quant=act_quant, nbits=nbits)
            m.mlp.up_proj = W8A8Linear.from_float(m.mlp.up_proj, weight_quant=weight_quant, act_quant=act_quant, nbits=nbits)
        elif "FalconDecoderLayer" in m.__class__.__name__:
            m.self_attention.query_key_value = W8A8Linear.from_float(
                m.self_attention.query_key_value, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input, nbits=nbits)
            m.self_attention.dense = W8A8Linear.from_float(m.self_attention.dense, weight_quant=weight_quant, act_quant=act_quant, nbits=nbits)
            m.mlp.dense_h_to_4h = W8A8Linear.from_float(m.mlp.dense_h_to_4h, weight_quant=weight_quant, act_quant=act_quant, nbits=nbits)
            m.mlp.dense_4h_to_h = W8A8Linear.from_float(m.mlp.dense_4h_to_h, weight_quant=weight_quant,
                                                        act_quant=act_quant, nbits=nbits)

    return module