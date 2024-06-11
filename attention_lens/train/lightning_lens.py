from __future__ import annotations

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
#import transformer_lens as tlens
#from transformer_lens import HookedTransformer

from attention_lens.lens import Lens
from attention_lens.model.get_model import get_model



class LightningLens(pl.LightningModule):
    def __init__(
        self,
        model_name: str,  # TODO: Add support for custom ``HookedTransformers``.
        lens_cls: type[Lens] | str,
        layer_num: int,
        lr: float = 1e-3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.model, self.tokenizer = get_model(model_name=self.model_name, device=self.device)
        if isinstance(lens_cls, str):
            lens_cls = Lens.get_lens(lens_cls)
        if isinstance(lens_cls, Lens):
            raise ValueError(
                "Argument `lens_cls` cannot be an instance of ``Lens`` class. Must be the class itself or a "
                "string corresponding to the class (check the ``Lens.registry``). Available ``Lens`` objects are:"
                f"{list(Lens.registry.keys())}"
            )
        
        #print(self.model.config.bos_token_id)
        self.layer_num = layer_num

        self.attn_lens = lens_cls(
            unembed=self.model.lm_head.weight,
            bias=self.model.transformer.h[self.layer_num].attn.c_proj.bias,
            n_head=self.model.config.num_attention_heads,
            d_model=self.model.config.hidden_size,
            d_vocab=self.model.config.vocab_size,
        )

        self.activation = {}
        self.register_hooks(self.model, self.get_activation)

        #self.hook_name = "result"
        self.lr = lr
        #self.hook_id = tlens.utils.get_act_name(self.hook_name, self.layer_num)
        #self.attention_layer = self.activation['layer_{self.layer_num}_attn'][1]

    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output[1]
        return hook
    
    def register_hooks(self, model, hook_func):
        for idx, layer in enumerate(model.transformer.h):
            layer.attn.register_forward_hook(hook_func(f"layer_{idx}_attn"))

    def kl_loss(self, logits, lens_logits) -> torch.Tensor:
        kldiv = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        k_logits, k_lens_out = F.log_softmax(logits[:, -1, :], dim=-1), F.log_softmax(
            lens_logits[:, -1, :], dim=-1
        )

        loss = kldiv(k_lens_out, k_logits)
        return loss

    ######################################################################################################

    def setup(self, stage) -> None:
        # TODO: There was a concern about how models were trained in distributed systems. Lightning does some
        #       additional setup in this setting, but `__init__` is only called on the master CPU. So, `self.model`
        #       and `self.hooked_model` are separate desppite being initialized identically. We need to confirm if
        #       they must be named differently for Lightning to work.
        self.model, self.tokenizer = get_model(
            model_name=self.model_name, device=self.trainer.strategy.root_device
        )

    def forward(self, cache: dict[str, torch.Tensor]) -> torch.Tensor:
        """

        Args:
            cache (dict[str, torch.Tensor]): A mapping from output results to a Tensor (part of ``HookedTransformer``).

        Returns:

        """
        inputs = list()
        #inputs.append(cache[self.hook_id])
        inputs.append(cache)
        inputs = torch.stack(inputs)[-1]
        # TODO: Double check that we need to pass in the LAST token position.
        return self.attn_lens(inputs)

    def training_step(self, train_batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        prompt = train_batch["text"]
        #tokens = self.model.to_tokens(prompt)

        prompt_tokens = self.tokenizer(prompt, return_tensors = 'pt', padding=True, truncation=True)
        prompt_length = prompt_tokens['input_ids'].shape[1]
        max_length = prompt_length + 1

        #NOTE: Had to use 'padding = true'
        inputs = self.tokenizer(prompt, max_length=max_length, truncation=True, padding=True, return_tensors='pt')

        # with torch.no_grad():
        #     # only cache required hooks for lens
        #     logits, cache = self.model.run_with_cache(
        #         tokens, names_filter=self.hook_id, remove_batch_dim=False
        #     )

        with torch.no_grad():
            self.activation.clear()
            outputs = self.model(**inputs)
            cache = self.activation['layer_{self.layer_num}_attn'][1]
            logits = outputs.logits

        lens_logits = self.forward(cache)
        loss = self.kl_loss(logits, lens_logits)
        self.log("train_loss", loss, prog_bar=True)
        # my_dict = {"train_loss":loss}
        # wandb.log(my_dict, step=trainer.global_step)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    # TODO(MS): register an early stopping call back which quits training if the loss/some metric drops below a certain pont
    # TODO(MS): when training quits, save a copy of the appropriately named lens
    # TODO(MS): test and make sure distributed training works accross nodes
