import copy

import torch
import torch.nn as nn
from jsq.fake_quant import W8A8Linear

def clip_matrix(matrix, abs=True, clip_l=0, clip_h=0, channel=False):
    if clip_l == 0 and clip_h == 0:
        return matrix

    if channel:
        print("Channel wise clip!")
        matrix_flatten = torch.t(matrix[0])
        if abs:
            matrix_flatten = torch.abs(torch.t(matrix[0]))
        max_threshold = None
        min_threshold = None

        if clip_h != 0:
            max_threshold = torch.quantile(matrix_flatten.double(), q=1 - clip_h, dim=0)
        clipped_matrix = torch.clamp(torch.t(matrix[0]), min=-max_threshold, max=max_threshold)
        return torch.t(clipped_matrix).unsqueeze(0).half()
    else:
        num_elements = matrix.numel()
        if abs:
            matrix_flatten = torch.abs(matrix).flatten()
        else:
            matrix_flatten = matrix.flatten()

        max_threshold = None
        min_threshold = None

        if clip_l != 0:
            low_index = int(clip_l * num_elements)
            min_threshold, _ = torch.topk(matrix_flatten, largest=False, k=low_index)
            min_threshold = min_threshold[-1]
        if clip_h != 0:
            high_index = int(clip_h * num_elements)
            max_threshold, _ = torch.topk(matrix_flatten, largest=True, k=high_index)
            max_threshold = max_threshold[-1]

        if abs:
            clipped_matrix = torch.clamp(matrix, -max_threshold, max_threshold)
        else:
            clipped_matrix = torch.clamp(matrix, min_threshold, max_threshold)

        return clipped_matrix


def find_layers(module, layers=[nn.Linear, W8A8Linear], name=''):
    if type(module) in layers or "FalconLinear" in module.__class__.__name__:
        return {name: module}
    else:
        pass
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def prepare_calibration_input(model, dataloader, device):
    CHATGLM = False
    Falcon = False
    if hasattr(model, "transformer"):
        if hasattr(model.transformer, "embedding"):
            CHATGLM = True
        elif hasattr(model.transformer, "word_embeddings"):
            Falcon = True
    use_cache = model.config.use_cache
    model.config.use_cache = False

    if CHATGLM:
        layers = model.transformer.encoder.layers
    elif Falcon:
        layers = model.transformer.h
    else:
        layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device) # (128, 2048, 4096)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, *args, **kwargs):
            if CHATGLM:
                inps[cache['i']] = inp.transpose(0, 1)[0]
            else:
                inps[cache['i']] = inp
            cache['i'] += 1
            if CHATGLM:
                cache['attention_mask'] = args[0]
            else:
                cache['attention_mask'] = kwargs['attention_mask']
            if CHATGLM:
                cache['position_ids'] = args[1]
            elif Falcon:
                pass
            else:
                cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    model.config.use_cache = use_cache
    return inps, outs, attention_mask, position_ids



def generate_ss(activation, weight):
    cin, cout = weight.shape
    ss = torch.zeros_like(weight)
    for i in range(cout):
        w = copy.deepcopy(weight)
        w[:, i] = 0
        out = activation @ (w.t())
        max_values, _ = torch.max(out, dim=0)
        min_values, _ = torch.min(out, dim=0)
        row_ss = (max_values - min_values)
        ss[:, i] = row_ss
    ss = torch.where(torch.isinf(ss), torch.tensor(100), ss)
    return ss
