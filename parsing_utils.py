from omegaconf import DictConfig, OmegaConf


def make_omegaconf_resolvers():
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
