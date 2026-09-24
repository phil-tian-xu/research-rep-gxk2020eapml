from utils.config_utils.config import get_cfg
from pathlib import Path
from utils.path_info import config_path
import pandas as pd


PROJECT_PREFIX = "[quant-datahub] Info:"
PROJECT_ERROR_PREFIX = "[quant-datahub] Error:"


def load_config(config_name):
    cfg = get_cfg()
    cfg.merge_from_file(Path(config_path, config_name + ".yaml"))
    return cfg

