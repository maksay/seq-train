from data_providers.base_data_provider import BaseDataProvider
from data_providers.duke_data_provider import DukeDataProvider

__all_data_providers__ = [
    "BaseDataProvider",
    "DukeDataProvider"
]


def make_data_provider(config):
    name = config["name"] if type(config) is dict else config.name
    if name in __all_data_providers__:
        return globals()[name](config)
    else:
        raise Exception('The data provider name %s does not exist' % name)

