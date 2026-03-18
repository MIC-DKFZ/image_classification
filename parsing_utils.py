from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from uuid import uuid4


def _get_model_embed_dim(model_type: str):
    if "dinov2" in model_type:
        dinov2_size_lookup = {"vits": 384, "vitb": 768, "vitl": 1024, "vitg": 1536}
        for type, dim in dinov2_size_lookup.items():
            if type in model_type:
                return dim
        raise ValueError(
            f"Invalid dinov2 model type '{model_type}'. Available ViT sizes: "
            f"{', '.join(dinov2_size_lookup.keys())}"
        )
    return None


def make_omegaconf_resolvers():
    def _make_group_name(model, ft_method):
        return (
            datetime.now().strftime("%Y%m%d_%H%M%S")
            + f"_{model.lower()}_{ft_method.lower()}_"
            + str(uuid4())
        )

    def _run_name_or_generated(run_name, model, ft_method):
        if run_name not in (None, "", "null"):
            return str(run_name)
        return _make_group_name(model, ft_method)

    OmegaConf.register_new_resolver(
        "path_formatter",
        lambda s: s.replace("[", "")
        .replace("]", "")
        .replace("}", "")
        .replace("{", "")
        .replace(")", "")
        .replace("(", "")
        .replace(",", "_")
        .replace("=", "_")
        .replace("/", ".")
        .replace("+", "")
        .replace("@", "."),
    )
    OmegaConf.register_new_resolver("model_name_extractor", lambda s: s.split(".")[-1])
    OmegaConf.register_new_resolver(
        "make_group_name",
        _make_group_name,
        use_cache=True,
    )
    OmegaConf.register_new_resolver(
        "run_name_or_generated",
        _run_name_or_generated,
        use_cache=True,
    )
    OmegaConf.register_new_resolver("group_extractor", lambda s: s.split("/")[-1])
    OmegaConf.register_new_resolver(
        "model_embed_dim_extractor", _get_model_embed_dim, use_cache=True
    )
