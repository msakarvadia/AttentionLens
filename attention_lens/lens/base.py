import torch.nn as nn
import torch


class Lens(nn.Module):
    """
    It is important to remember that subclasses of the ``Lens`` cannot take separate arguments as they
    will be ignored at runtime for instantiation in the ``TransformerAttnInterface``.
    """

    registry = {}

    def __init__(self, unembed, bias, n_head, d_model, d_vocab) -> None:
        """

        Args:
            unembed ():
            bias ():
            n_head ():
            d_model ():
            d_vocab ():
        """
        super().__init__()
        self.unembed = unembed
        self.bias = bias
        self.n_head = n_head
        self.d_model = d_model
        self.d_vocab = d_vocab

    @classmethod
    def get_lens(cls, name: str) -> type["Lens"]:
        """
        This takes the name of the lens and queries the registry to grab the corresponding ``Lens`` subclas.

        Args:
            name (str): The name of the child ``Lens`` implementation.

        Returns:
            Subclass ``Lens`` implementation.
        """
        name = name.lower()
        if name in cls.registry:
            return cls.registry[name]
        else:
            raise KeyError(
                f"Strategy name ({name=}) is not in the Strategy registry. Available ``Lens`` objects are:"
                f"{list(cls.registry.keys())}"
            )

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls.registry[cls.__name__.lower()] = cls


'''
class TransformerAttentionLense(nn.Module):
    # NOTE (MS): this lens can support many layers
    # but we will only create one lens per layer at a time
    # to reduce the amount of memory we use
    # The attnlense for all layers for gpt2_small is roughly ~5B parameters
    # 768*50,000 * 12 * 12 (d_model* d_vocab* n_head* n_layer)
    def __init__(
        self,
        unembed,
        bias,
        n_layers,
        n_head: int,
        d_model: int,
        d_vocab: int,
        lense_cls: type[Lens] = LenseA,
    ):
        super().__init__()
        self.lenses = nn.ModuleList(
            [
                lense_cls(unembed, bias, n_head, d_model, d_vocab)
                for i in range(n_layers)
            ]
        )

    # accepts a tensor of dims [n_layers, <hook_attn_result_size...>]
    def forward(self, x):
        outs = []
        for i in range(self.n_layers):
            o = self.lenses[i](x[i])
            outs.append(o)

        return torch.stack(outs)

    @property
    def n_layers(self) -> int:
        return len(self.lenses)


def get_lense(
    unembed, bias, n_layers=1, n_head=12, d_model=768, d_vocab=50257, lense_class=LenseA
):
    """This is a description

    You can even do this, $f(x)=x^2$

    Args:
        unembed (int): This is the unembedding matrix
        bias (float): _description_
        n_layers (int, optional): _description_. Defaults to 1.
        n_head (int, optional): _description_. Defaults to 12.
        d_model (int, optional): _description_. Defaults to 768.
        d_vocab (int, optional): _description_. Defaults to 50257.
        lense_class (cls, optional): _description_. Defaults to LenseA.

    Examples:
        >>> lens = get_lense(...)

    Returns:
        lens: This is the hooked lens!
    """
    lens = TransformerAttentionLense(
        unembed, bias, n_layers, n_head, d_model, d_vocab, lense_class=lense_class
    )
    return lens
'''

