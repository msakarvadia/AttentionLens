import torch.nn as nn
import torch
from attention_lense.lense.lenseA import LenseA

class TransformerAttentionLense(nn.Module):
    #NOTE (MS): this lense can support many layers
    # but we will only create one lense per layer at a time
    # to reduce the amount of memory we use
    # The attnlense for all layers for gpt2_small is roughly ~5B parameters
    # 768*50,000 * 12 * 12 (d_model* d_vocab* n_head* n_layer) 
    def __init__(self, unembed, bias, n_layers, n_head, d_model, d_vocab, lense_class=LenseA):
        super().__init__()
        self.n_layers = n_layers
        self.lenses = nn.ModuleList([lense_class(unembed, bias, n_head, d_model, d_vocab) for i in range(n_layers)])

    # accepts a tensor of dims [n_layers, <hook_attn_result_size...>]
    def forward(self, x):
        outs = []
        for i in range(self.n_layers):
            o = self.lenses[i](x[i])
            outs.append(o)

        return torch.stack(outs)

def get_lense(unembed, bias,  n_layers=1, n_head=12, d_model=768, d_vocab=50257, lense_class=LenseA):
    lense =  TransformerAttentionLense(unembed,
                                       bias,
                                       n_layers,
                                       n_head,
                                       d_model,
                                       d_vocab, 
                                       lense_class = lense_class)
    return lense

