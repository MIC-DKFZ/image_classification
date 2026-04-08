from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def sparse_topk_attention(
    attention_logits: torch.Tensor,
    *,
    k: int = 2,
    tau: float = 0.25,
    stochastic: bool = True,
    training: bool = True,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sparse top-k attention weights from raw attention logits."""
    original_dim = attention_logits.dim()
    if original_dim == 1:
        logits = attention_logits.unsqueeze(0).unsqueeze(0)
    elif original_dim == 2:
        logits = attention_logits.unsqueeze(0)
    elif original_dim == 3:
        logits = attention_logits
    else:
        raise ValueError(
            f"sparse_topk_attention expects 1D/2D/3D attention, got shape={tuple(attention_logits.shape)}"
        )

    mask_3d = _normalize_attention_mask(mask, logits)
    if stochastic and training:
        uniform = torch.rand_like(logits).clamp_min(1e-9)
        logits = logits + (-torch.log(-torch.log(uniform)))
    logits = logits / float(tau)
    weights = _masked_softmax(logits, mask_3d)

    if mask_3d is not None:
        valid_counts = mask_3d.sum(dim=-1).clamp_min(1)
        max_k = int(valid_counts.min().item())
    else:
        max_k = logits.shape[-1]
    k_eff = max(1, min(int(k), max_k))
    topk_idx = weights.topk(k_eff, dim=-1).indices
    topk_mask = torch.zeros_like(weights).scatter_(-1, topk_idx, 1.0)
    sparse = weights * topk_mask
    sparse = sparse / (sparse.sum(dim=-1, keepdim=True) + eps)

    if original_dim == 1:
        return sparse.squeeze(0).squeeze(0), topk_idx.squeeze(0).squeeze(0)
    if original_dim == 2:
        return sparse.squeeze(0), topk_idx.squeeze(0)
    return sparse, topk_idx


class CosineClassifier(nn.Module):
    def __init__(self, dim: int, num_classes: int, scale: float = 20.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, dim))
        self.scale = float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, dim=-1)
        weight = F.normalize(self.weight, dim=-1)
        return self.scale * (x @ weight.t())


class CosineBinaryHead(nn.Module):
    def __init__(self, dim: int, scale: float = 20.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, dim))
        self.scale = float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, dim=-1)
        weight = F.normalize(self.weight, dim=-1)
        return self.scale * (x @ weight.t())


class FeaturePrep(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, l2_normalize: bool = True):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim, eps=eps)
        self.l2_normalize = bool(l2_normalize)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        if self.l2_normalize:
            x = F.normalize(x, p=2, dim=-1)
        return x


class AttnNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, n_classes: int):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.net(x), x


class GatedAttnNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, n_classes: int):
        super().__init__()
        a_layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        b_layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.Sigmoid()]
        if dropout > 0:
            a_layers.append(nn.Dropout(dropout))
            b_layers.append(nn.Dropout(dropout))
        self.attention_a = nn.Sequential(*a_layers)
        self.attention_b = nn.Sequential(*b_layers)
        self.attention_c = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.attention_a(x)
        b = self.attention_b(x)
        return self.attention_c(a * b), x


def _normalize_attention_mask(mask: torch.Tensor | None, attention: torch.Tensor) -> torch.Tensor | None:
    if mask is None:
        return None
    if mask.dtype != torch.bool:
        mask = mask.bool()
    if attention.dim() != 3:
        raise ValueError(f"Expected 3D attention for mask normalization, got {attention.dim()}D.")
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    if mask.dim() != 2:
        raise ValueError(f"Expected 1D/2D mask, got shape={tuple(mask.shape)}")
    if mask.shape[0] != attention.shape[0] or mask.shape[1] != attention.shape[2]:
        raise ValueError(
            f"Mask shape {tuple(mask.shape)} is incompatible with attention shape {tuple(attention.shape)}."
        )
    return mask.unsqueeze(1).expand(-1, attention.shape[1], -1)


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return F.softmax(logits, dim=-1)
    masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    weights = F.softmax(masked_logits, dim=-1)
    weights = weights * mask.to(dtype=weights.dtype)
    denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return weights / denom


def _attention_dropout_mask(
    attention_logits: torch.Tensor,
    *,
    p_drop: float,
    training: bool,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if not training or p_drop <= 0:
        return None

    dropout_mask = torch.rand_like(attention_logits).gt(p_drop)
    if valid_mask is not None:
        dropout_mask = dropout_mask & valid_mask
        top_source = attention_logits.masked_fill(~valid_mask, torch.finfo(attention_logits.dtype).min)
    else:
        top_source = attention_logits
    top_idx = top_source.argmax(dim=-1, keepdim=True)
    dropout_mask.scatter_(-1, top_idx, True)
    return dropout_mask


def _select_topk(scores: torch.Tensor, k: int, *, largest: bool, valid_mask: torch.Tensor | None) -> torch.Tensor:
    if valid_mask is not None:
        fill_value = torch.finfo(scores.dtype).min if largest else torch.finfo(scores.dtype).max
        scores = scores.masked_fill(~valid_mask, fill_value)
        valid_count = int(valid_mask.sum().item())
    else:
        valid_count = int(scores.numel())
    if valid_count <= 0:
        raise ValueError("Cannot select top-k instances from an empty bag.")
    return torch.topk(scores, k=min(k, valid_count), largest=largest).indices


def _unpack_bag_input(x: torch.Tensor | dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(x, dict):
        if "features" not in x:
            raise KeyError("MIL bag input dict must contain a 'features' entry.")
        features = x["features"]
        mask = x.get("mask")
    else:
        features = x
        mask = None

    if features.ndim not in (2, 3):
        raise ValueError(
            f"CLAM expects a bag tensor with shape [N, D] or [B, N, D], got {tuple(features.shape)}."
        )

    if mask is not None:
        if mask.ndim == 1 and features.ndim == 2:
            return features, mask.bool()
        if mask.ndim == 2 and features.ndim == 3:
            return features, mask.bool()
        raise ValueError(
            f"Bag mask shape {tuple(mask.shape)} is incompatible with feature shape {tuple(features.shape)}."
        )
    return features, None


class _BaseClam(nn.Module):
    consumes_raw_features = True

    def __init__(
        self,
        *,
        gate: bool,
        size_arg: str,
        dropout: float,
        k_sample: int,
        n_classes: int,
        subtyping: bool,
        embed_dim: int,
        feature_prep: bool,
        l2_normalize_features: bool,
        layer_norm_eps: float,
        cosine_head: bool,
        cosine_scale: float,
        instance_eval: bool,
        instance_loss_weight: float,
        attn_drop: float,
        topk_k: int,
        topk_tau: float,
        stochastic_topk: bool,
    ):
        super().__init__()
        self.size_dict = {
            "tiny": [embed_dim, 256, 128],
            "small": [embed_dim, 512, 256],
            "big": [embed_dim, 512, 384],
        }
        if size_arg not in self.size_dict:
            raise ValueError(f"Unknown CLAM size_arg={size_arg!r}.")

        size = self.size_dict[size_arg]
        prep_layers: list[nn.Module] = []
        if feature_prep:
            prep_layers.append(
                FeaturePrep(
                    size[0],
                    eps=layer_norm_eps,
                    l2_normalize=l2_normalize_features,
                )
            )
        prep_layers.extend([nn.Linear(size[0], size[1]), nn.ReLU()])
        if dropout > 0:
            prep_layers.append(nn.Dropout(dropout))
        attention_cls = GatedAttnNet if gate else AttnNet
        prep_layers.append(
            attention_cls(
                input_dim=size[1],
                hidden_dim=size[2],
                dropout=dropout,
                n_classes=self._attention_branches(n_classes),
            )
        )
        self.attention_net = nn.Sequential(*prep_layers)
        self.instance_classifiers = nn.ModuleList(
            [nn.Linear(size[1], 2) for _ in range(n_classes)]
        )
        self.instance_loss_fn = nn.CrossEntropyLoss()
        self.k_sample = int(k_sample)
        self.n_classes = int(n_classes)
        self.subtyping = bool(subtyping)
        self.instance_eval_enabled = bool(instance_eval)
        self.instance_loss_weight = float(instance_loss_weight)
        self.attn_drop = float(attn_drop)
        self.k_topk = int(topk_k)
        self.attn_tau = float(topk_tau)
        self.stochastic_topk = bool(stochastic_topk)
        self.projected_dim = int(size[1])
        self.classifiers = self._build_classifiers(
            hidden_dim=size[1],
            n_classes=n_classes,
            cosine_head=cosine_head,
            cosine_scale=cosine_scale,
        )
        self.last_aux: dict[str, torch.Tensor] = {}

    @staticmethod
    def create_positive_targets(length: int, device: torch.device) -> torch.Tensor:
        return torch.ones(length, device=device, dtype=torch.long)

    @staticmethod
    def create_negative_targets(length: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(length, device=device, dtype=torch.long)

    def _attention_branches(self, n_classes: int) -> int:
        raise NotImplementedError

    def _build_classifiers(
        self,
        *,
        hidden_dim: int,
        n_classes: int,
        cosine_head: bool,
        cosine_scale: float,
    ) -> nn.Module:
        raise NotImplementedError

    def _bag_logits(self, bag_features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _prepare_attention(
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        features, mask = _unpack_bag_input(x)
        raw_attention, projected = self.attention_net(features)

        if raw_attention.dim() == 2:
            attention_logits = raw_attention.transpose(1, 0).unsqueeze(0)
            projected = projected.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)
        else:
            attention_logits = raw_attention.transpose(1, 2)

        valid_mask = _normalize_attention_mask(mask, attention_logits)
        keep_mask = _attention_dropout_mask(
            attention_logits,
            p_drop=self.attn_drop,
            training=self.training,
            valid_mask=valid_mask,
        )
        if keep_mask is not None:
            if valid_mask is not None:
                keep_mask = keep_mask & valid_mask
            attention_logits = attention_logits.masked_fill(
                ~keep_mask, torch.finfo(attention_logits.dtype).min
            )

        attention_weights = _masked_softmax(attention_logits, valid_mask)
        bag_features = torch.matmul(attention_weights, projected)
        if features.dim() == 2:
            bag_features = bag_features.squeeze(0)
            attention_weights = attention_weights.squeeze(0)
            attention_logits = attention_logits.squeeze(0)
            projected = projected.squeeze(0)
        return attention_logits, bag_features, projected, mask

    def _instance_eval_single_bag(
        self,
        attention_weights: torch.Tensor,
        projected_features: torch.Tensor,
        label: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        inst_labels = F.one_hot(label.view(-1), num_classes=self.n_classes).squeeze(0)
        total_inst_loss: torch.Tensor | None = None
        valid_mask = mask if mask is not None else None

        for class_index, classifier in enumerate(self.instance_classifiers):
            inst_label = int(inst_labels[class_index].item())
            branch_attention = attention_weights[min(class_index, attention_weights.shape[0] - 1)]

            if inst_label == 1:
                loss = self._positive_negative_instance_loss(
                    branch_attention,
                    projected_features,
                    classifier,
                    valid_mask=valid_mask,
                )
            elif self.subtyping:
                loss = self._out_of_class_instance_loss(
                    branch_attention,
                    projected_features,
                    classifier,
                    valid_mask=valid_mask,
                )
            else:
                continue
            total_inst_loss = loss if total_inst_loss is None else total_inst_loss + loss

        if total_inst_loss is None:
            return None
        if self.subtyping and len(self.instance_classifiers) > 0:
            total_inst_loss = total_inst_loss / len(self.instance_classifiers)
        return total_inst_loss

    def _positive_negative_instance_loss(
        self,
        attention: torch.Tensor,
        projected_features: torch.Tensor,
        classifier: nn.Module,
        *,
        valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        device = projected_features.device
        top_pos_idx = _select_topk(attention, self.k_sample, largest=True, valid_mask=valid_mask)
        top_neg_idx = _select_topk(attention, self.k_sample, largest=False, valid_mask=valid_mask)
        top_pos = torch.index_select(projected_features, dim=0, index=top_pos_idx)
        top_neg = torch.index_select(projected_features, dim=0, index=top_neg_idx)
        pos_targets = self.create_positive_targets(top_pos.shape[0], device)
        neg_targets = self.create_negative_targets(top_neg.shape[0], device)
        all_instances = torch.cat([top_pos, top_neg], dim=0)
        all_targets = torch.cat([pos_targets, neg_targets], dim=0)
        logits = classifier(all_instances)
        return self.instance_loss_fn(logits, all_targets)

    def _out_of_class_instance_loss(
        self,
        attention: torch.Tensor,
        projected_features: torch.Tensor,
        classifier: nn.Module,
        *,
        valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        device = projected_features.device
        top_idx = _select_topk(attention, self.k_sample, largest=True, valid_mask=valid_mask)
        top_features = torch.index_select(projected_features, dim=0, index=top_idx)
        neg_targets = self.create_negative_targets(top_features.shape[0], device)
        logits = classifier(top_features)
        return self.instance_loss_fn(logits, neg_targets)

    def get_aux_loss(self) -> torch.Tensor | None:
        return self.last_aux.get("loss")

    def forward(
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
        *,
        label: torch.Tensor | None = None,
        attention_only: bool = False,
    ) -> torch.Tensor:
        attention_logits, bag_features, projected_features, mask = self._prepare_attention(x)
        if attention_only:
            return attention_logits

        logits = self._bag_logits(bag_features)
        self.last_aux = {}
        if self.instance_eval_enabled and label is not None:
            if logits.dim() == 1:
                label = label.view(1)
            if attention_logits.dim() == 2:
                instance_loss = self._instance_eval_single_bag(
                    attention_weights=_masked_softmax(
                        attention_logits.unsqueeze(0),
                        _normalize_attention_mask(mask, attention_logits.unsqueeze(0)),
                    ).squeeze(0),
                    projected_features=projected_features,
                    label=label[0],
                    mask=mask,
                )
            else:
                batch_losses = []
                mask_batch = mask
                attention_weights = _masked_softmax(
                    attention_logits,
                    _normalize_attention_mask(mask, attention_logits),
                )
                for batch_idx in range(attention_logits.shape[0]):
                    batch_losses.append(
                        self._instance_eval_single_bag(
                            attention_weights=attention_weights[batch_idx],
                            projected_features=projected_features[batch_idx],
                            label=label[batch_idx],
                            mask=None if mask_batch is None else mask_batch[batch_idx],
                        )
                    )
                valid_losses = [loss for loss in batch_losses if loss is not None]
                instance_loss = None
                if valid_losses:
                    instance_loss = torch.stack(valid_losses).mean()
            if instance_loss is not None and self.instance_loss_weight > 0:
                self.last_aux["instance_loss"] = instance_loss
                self.last_aux["loss"] = instance_loss * self.instance_loss_weight

        return logits

    def forward_with_attn(
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
        *,
        return_topk: bool = False,
        k: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        attention_logits, bag_features, _, mask = self._prepare_attention(x)
        logits = self._bag_logits(bag_features)
        if not return_topk:
            return logits, attention_logits

        attention_topk, topk_indices = sparse_topk_attention(
            attention_logits,
            k=self.k_topk if k is None else k,
            tau=self.attn_tau,
            stochastic=self.stochastic_topk,
            training=self.training,
            mask=mask,
        )
        return logits, attention_logits, attention_topk, topk_indices

    def forward_for_heatmap(
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attention_logits, bag_features, _, _ = self._prepare_attention(x)
        logits = self._bag_logits(bag_features)
        return logits, attention_logits


class CLAM_SB(_BaseClam):
    def _attention_branches(self, n_classes: int) -> int:
        _ = n_classes
        return 1

    def _build_classifiers(
        self,
        *,
        hidden_dim: int,
        n_classes: int,
        cosine_head: bool,
        cosine_scale: float,
    ) -> nn.Module:
        if cosine_head:
            return CosineClassifier(hidden_dim, n_classes, scale=cosine_scale)
        return nn.Linear(hidden_dim, n_classes)

    def _bag_logits(self, bag_features: torch.Tensor) -> torch.Tensor:
        if bag_features.dim() == 3:
            bag_features = bag_features.squeeze(1)
        return self.classifiers(bag_features)


class CLAM_MB(_BaseClam):
    def _attention_branches(self, n_classes: int) -> int:
        return n_classes

    def _build_classifiers(
        self,
        *,
        hidden_dim: int,
        n_classes: int,
        cosine_head: bool,
        cosine_scale: float,
    ) -> nn.Module:
        if cosine_head:
            return nn.ModuleList(
                [CosineBinaryHead(hidden_dim, scale=cosine_scale) for _ in range(n_classes)]
            )
        return nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(n_classes)])

    def _bag_logits(self, bag_features: torch.Tensor) -> torch.Tensor:
        if bag_features.dim() == 2:
            bag_features = bag_features.unsqueeze(0)
        batch_size = bag_features.shape[0]
        logits = torch.empty(batch_size, self.n_classes, device=bag_features.device)
        for class_index, classifier in enumerate(self.classifiers):
            logits[:, class_index] = classifier(bag_features[:, class_index, :]).squeeze(-1)
        return logits
