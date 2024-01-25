"""
To see all the implemented ``Lens``, simply import ``Lens`` and print ``Lens.registry``.

```mermaid
flowchart LR
    Lens
    subgraph Registry
       LensA
    end

    Lens-->LensA
```
"""


from attention_lens.lens.base import Lens
from attention_lens.lens.registry.lensA import LensA

__all__ = ["Lens", "LensA"]
