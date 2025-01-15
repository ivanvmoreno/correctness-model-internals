from typing import Text

from box import ConfigBox


def load_config(config_path: Text) -> ConfigBox:
    """Loads yaml config in instance of box.ConfigBox.
    Args:
        config_path {Text}: path to config
    Returns:
        box.ConfigBox
    """
    config = ConfigBox.from_yaml(filename=config_path)
    return config
