import sys
sys.path.append('../')
from data.get_data import get_data
from model.get_model import get_model
from lense.get_lense import get_lense
import torch
import transformer_lens.utils as utils
import torch.nn.functional as F


device = "cuda:0" if torch.cuda.is_available() else "cpu"
pin_memory=False
if device!="cpu":
    pin_memory=True

dataloader = get_data(batch_size=2, pin_memory=pin_memory, device=device)

model = get_model(device=device)


for data in dataloader:
    print(model(data['text']))
    print(data['text'])
    break

# do the training one layer at a time to prevent ram from running out
hook_name = 'result'
n_layers = 12
kldiv = torch.nn.KLDivLoss(reduction='batchmean')

for n_layer in range(n_layers):
  print("layer: ", n_layer)

  #TODO: make lens w/ parameters
  #lens_param = {'n_head': model.cfg.n_heads, 'd_model': model.cfg.d_model, 'd_vocab': model.cfg.d_vocab, 'lens_cls': LensA}
  #attn_lens = TransformerAttentionLense(1, **lens_param).to(device)
  attn_lens = get_lense().to(device)

  #todo: need to enable batching
  for data in dataloader:
    prompt = data['text']
  #for prompt in plain_data:
    print(prompt)

    # hook_id = utils.get_act_name(hook_name, 9)
    tokens = model.to_tokens(prompt)
    print("token size: ", tokens.size())

    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens, remove_batch_dim=False)
    
    inputs = []
    hook_id = utils.get_act_name(hook_name, n_layer)
    inputs.append(cache[hook_id])

    input_tensor = torch.stack(inputs)

    # print(input.requires_grad, input.grad_fn)f

    # lens_out = lens(input)
    # print(lens_out.requires_grad)
    # print(lens_out.grad_fn)

    #print("input size: ",input_tensor.size())

    attn_lens_out = attn_lens(input_tensor)

    #print(attn_lens_out.size())

    lens_out = attn_lens_out[0]
    #print(lens_out.size())

    k_logits, k_lens_out = F.log_softmax(logits, dim=-1), F.log_softmax(lens_out, dim=-1)

    #print(k_lens_out.requires_grad)
    #print(k_lens_out.grad_fn)

    loss = kldiv(k_lens_out, k_logits)
    loss.backward()
    
  #TODO: need to save attn lens in correct location
  name = "attn_lens_layer_"+str(n_layer)
  torch.save(attn_lens, name)




