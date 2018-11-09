from training.managers import VirtualManager
from utils.util import parse_config

CONFIG_FILENAME = 'config.json'
AFTERLEARN_CONFIG_FILENAME = 'afterlearn_config.json'


if __name__ == '__main__':
    config = parse_config(config_name=CONFIG_FILENAME)
    manager = VirtualManager(config)
    manager.manage()
