import torch.nn as nn
import torch
from lense.lenseA import LenseA

class TransformerAttentionLense(nn.Module):
    #TODO: clean this up, and make it per layer by default 
    def __init__(self, n_layers, n_head, d_model, d_vocab, lense_class=LenseA):
        super().__init__()
        self.n_layers = n_layers
        self.lenses = nn.ModuleList([lense_class(n_head, d_model, d_vocab) for i in range(n_layers)])

    # accepts a tensor of dims [n_layers, <hook_attn_result_size...>]
    def forward(self, x):
        outs = []
        for i in range(self.n_layers):
            o = self.lenses[i](x[i])
            outs.append(o)

        return torch.stack(outs)

def get_lense(n_layers=1, n_head=12, d_model=768, d_vocab=50257, lense_class=LenseA):
    lense =  TransformerAttentionLense(n_layers,
                                       n_head,
                                       d_model,
                                       d_vocab, 
                                       lense_class = lense_class)
    return lense

