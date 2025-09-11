from __future__ import annotations
import math
from typing import Iterable, Optional, Sequence, Tuple, List

import torch
from torch import nn


class VisualPromptTuning:
    """
    Visual Prompt Tuning (VPT) wrapper that augments an existing ViT/EVA model
    by prepending learnable prompt tokens (shallow or deep) via forward pre-hooks.

    - No edits to the backbone.
    - Shallow VPT: insert prompts before the first block once; they flow through all blocks.
    - Deep VPT: for each block i, remove previous prompts and insert a new, per-layer prompt.

    Args:
        model:              Pretrained ViT/EVA backbone (timm-style).
        num_tokens:         Number of prompt tokens to prepend.
        deep:               If True, use VPT-Deep; else VPT-Shallow.
        project_dim:        If not None and != hidden_dim, project prompts to `hidden_dim`.
        dropout:            Dropout applied to prompt tokens.
        init_scale:         Xavier-uniform range scale (matches common VPT init).
        blocks_attr:        Attribute name holding the transformer blocks (default: "blocks").
        hidden_dim:         Force hidden size if it cannot be inferred.
        deep_layers:        Optional subset of layer indices to apply deep prompts to.
                            If None and deep=True, applies to all layers.
    """

    def __init__(
        self,
        num_tokens: int = 20,
        deep: bool = False,
        project_dim: Optional[int] = None,
        dropout: float = 0.0,
        init_scale: float = 1.0,
        blocks_attr: str = "blocks",
        hidden_dim: Optional[int] = None,
        deep_layers: Optional[Sequence[int]] = None,
        *args, **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.blocks_attr = blocks_attr
        self.num_tokens = int(num_tokens)
        self.deep = bool(deep)
        self.dropout = nn.Dropout(dropout)

        # Locate transformer blocks (timm/EVA style)
        if not hasattr(self.model.vit.encoder, self.blocks_attr):
            raise AttributeError(
                f"Backbone has no '{self.blocks_attr}' attribute. "
                "Pass the correct attribute via `blocks_attr`."
            )
        self.blocks = getattr(self.model.vit.encoder, self.blocks_attr)
        if not isinstance(self.blocks, (nn.Sequential, nn.ModuleList)) or len(self.blocks) == 0:
            raise ValueError("Expected a non-empty nn.Sequential or nn.ModuleList of blocks.")

        # Hidden size
        self.hidden_dim = int(hidden_dim) if hidden_dim is not None else self.model.vit.encoder.layer[0].layernorm_before.normalized_shape[0]

        # Prompt (content) dimension before optional projection
        prompt_in_dim = self.hidden_dim if project_dim is None else int(project_dim)

        # Prompt parameters
        # Xavier-like uniform range (matches common VPT init heuristic)
        val = (init_scale * math.sqrt(6.0 / (3 * 16 * 16 + prompt_in_dim)))  # 3 * 16*16 = rough RGB*patch_area; harmless for init
        self.prompt_embeddings = nn.Parameter(torch.empty(1, self.num_tokens, prompt_in_dim))
        nn.init.uniform_(self.prompt_embeddings, -val, val)

        # Optional per-layer deep prompts
        self.deep_layers: List[int]
        if self.deep:
            all_idx = list(range(len(self.blocks)))
            self.deep_layers = list(deep_layers) if deep_layers is not None else all_idx
            # Typically VPT-Deep uses prompts for all but possibly the last; using all is fine.
            self.deep_prompt_embeddings = nn.Parameter(
                torch.empty(len(self.deep_layers), self.num_tokens, prompt_in_dim)
            )
            nn.init.uniform_(self.deep_prompt_embeddings, -val, val)
        else:
            self.register_parameter("deep_prompt_embeddings", None)
            self.deep_layers = []

        # Optional projection to transformer hidden size
        if project_dim is not None and project_dim != self.hidden_dim:
            self.prompt_proj = nn.Linear(prompt_in_dim, self.hidden_dim)
            nn.init.kaiming_normal_(self.prompt_proj.weight, a=0.0, mode="fan_out")
        else:
            self.prompt_proj = nn.Identity()

        # Hook handles
        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self._install_hooks()

        self.freeze_backbone()

    # ---------- Public control ----------

    def freeze_backbone(self, eval_mode: bool = True) -> None:
        """Freeze all backbone params; keep only prompts trainable."""
        for p in self.model.parameters():
            p.requires_grad = False
        # Ensure prompt params are trainable
        for p in self.prompt_parameters():
            p.requires_grad = True
        if eval_mode:
            self.model.eval()

    def unfreeze_backbone(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = True
        self.model.train()

    def prompt_parameters(self) -> Iterable[nn.Parameter]:
        yield self.prompt_embeddings
        if isinstance(self.prompt_proj, nn.Linear):
            yield from self.prompt_proj.parameters()
        if self.deep and self.deep_prompt_embeddings is not None:
            yield self.deep_prompt_embeddings

    def remove_hooks(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    # ---------- Internal: hooks ----------

    def _install_hooks(self) -> None:
        self.remove_hooks()
        if self.deep:
            # Per-layer hooks
            idx_map = {layer_idx: i for i, layer_idx in enumerate(self.deep_layers)}
            for layer_idx, block in enumerate(self.blocks):
                if layer_idx in idx_map:
                    dp_index = idx_map[layer_idx]
                    h = block.register_forward_pre_hook(
                        self._make_deep_hook(dp_index), with_kwargs=False
                    )
                    self._hook_handles.append(h)
        else:
            # Shallow: only before the first block
            first_block = self.blocks[0]
            h = first_block.register_forward_pre_hook(self._make_shallow_hook(), with_kwargs=False)
            self._hook_handles.append(h)

    def _make_shallow_hook(self):
        """
        Insert the (same) prompt tokens once before block 0; they then flow through
        all later blocks unchanged. This mimics shallow VPT: prepend after [CLS].
        """
        def hook(_module: nn.Module, inputs: Tuple[torch.Tensor, ...]):
            x = inputs[0]  # x: [B, 1+N, D] after pos_embed + drop
            B = x.shape[0]
            pe = self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)
            pe = self.dropout(pe)
            # Insert after CLS (index 0): [CLS] + [PROMPTS] + [PATCHES]
            x_new = torch.cat((x[:, :1, :], pe, x[:, 1:, :]), dim=1)
            inputs = (x_new, *inputs[1:])
            return inputs
        return hook

    def _make_deep_hook(self, dp_index: int):
        """
        For block k: remove previous prompts (positions 1..P if present), then
        insert this block's deep prompts after CLS. Matches common VPT-Deep.
        """
        def hook(_module: nn.Module, inputs: Tuple[torch.Tensor, ...]):
            x = inputs[0]  # x: [B, 1+N(+P), D]
            B = x.shape[0]
            # Remove previously injected prompts if sequence already contains them
            if x.shape[1] >= 1 + self.num_tokens + 1:
                # assume prompts occupy positions 1..P
                x = torch.cat((x[:, :1, :], x[:, 1 + self.num_tokens :, :]), dim=1)

            pe = self.prompt_proj(self.deep_prompt_embeddings[dp_index]).expand(B, -1, -1)
            pe = self.dropout(pe)
            x_new = torch.cat((x[:, :1, :], pe, x[:, 1:, :]), dim=1)
            inputs = (x_new, *inputs[1:])
            return inputs
        return hook
