import yaml
from pathlib import Path
from pcc.params import TrainConfig, PCCConfig, HYPERPARAMS
from pcc.train.train import train


class Loader(yaml.SafeLoader):
    pass


def include(loader, node):
    filename = loader.construct_scalar(node)
    base_path = Path(__file__).parent.parent
    with open(base_path / filename, "r") as f:
        return yaml.load(f, Loader)


Loader.add_constructor("!include", include)


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader)

    return config


def get_train_config(config_path):
    config_dict = load_config(config_path)
    pcc_params = config_dict.get("pcc", {})
    hyp_params = config_dict.get("hyperparams", {})
    train_params = config_dict.get("train", {})

    config = TrainConfig
    config.pcc = PCCConfig(**pcc_params)
    config.hyperparams = HYPERPARAMS(**hyp_params)
    for key, value in train_params.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


if __name__ == "__main__":
    config = get_train_config("experiments/exp_00.yaml")
    train(config)
