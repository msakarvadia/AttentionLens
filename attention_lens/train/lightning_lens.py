from __future__ import annotations

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
import transformers

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

        if self.model.lm_head.bias == None:
            self.bias = torch.zeros(self.model.config.vocab_size).to(self.device)

        self.attn_lens = lens_cls(
            unembed=self.model.lm_head.weight.T,
            bias=self.bias,
            n_head=self.model.config.num_attention_heads,
            d_model=self.model.config.hidden_size,
            d_vocab=self.model.config.vocab_size,
        )

        self.hook_name = "result"
        self.layer_num = layer_num
        self.lr = lr
        #self.hook_id = tlens.utils.get_act_name(self.hook_name, self.layer_num)

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
            model_name=self.model_name, 
            device=self.trainer.strategy.root_device,
        )

    def forward(self, cache) -> torch.Tensor:
        """

        Args:
            cache (dict[str, torch.Tensor]): A mapping from output results to a Tensor (part of ``HookedTransformer``).

        Returns:

        """
        inputs = list()
        inputs.append(cache)
        inputs = torch.stack(inputs)[-1]
        # TODO: Double check that we need to pass in the LAST token position.
        return self.attn_lens(inputs)

    def training_step(self, train_batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        prompt = train_batch["text"]
        #tokens = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)
        inputs = self.tokenizer(
            prompt,
            truncation = True,
            padding = True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            cache = self.model.transformer.h[self.layer_num].attn.head_out
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
