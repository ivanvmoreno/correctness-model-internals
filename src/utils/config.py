from typing import Text

from dotenv import load_dotenv
from box import ConfigBox

load_dotenv()


def load_config(config_path: Text) -> ConfigBox:
    """Loads yaml config in instance of box.ConfigBox.
    Args:
        config_path {Text}: path to config
    Returns:
        box.ConfigBox
    """
    config = ConfigBox.from_yaml(filename=config_path)
    return config
