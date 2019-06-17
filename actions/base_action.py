import logging


class BaseAction(object):
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("main.actions.%s" % config["name"])

        self.logger.info("Creating Action:")
        for k, v in self.config.items():
            setattr(self, k, v)
            self.logger.info("%20s: %s", k, v)

    def do(self, **kwargs):
        raise NotImplementedError()

