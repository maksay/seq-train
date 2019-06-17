import os
import tensorflow as tf
import logging


class BaseModel(object):
    def __init__(self, config, mode, sess, ckpt_dir):
        self._saver = None
        self.sess = sess

        self.config = config
        self.logger = logging.getLogger("main.models.%s" % config["name"])

        for k, v in self.config.items():
            setattr(self, k, v)
            self.logger.info("%20s: %s", k, v)

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.ckpt_dir = ckpt_dir

    def save_model(self, custom_dir=None):
        self.logger.info(" [*] Saving checkpoints...")

        ckpt_dir = self.ckpt_dir if custom_dir is None else custom_dir

        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        self.saver.save(self.sess, os.path.join(ckpt_dir, "model"),
                        global_step=self.global_step)

    def load_model(self, custom_dir=None):
        self.logger.info(" [*] Loading checkpoints...")

        ckpt_dir = self.ckpt_dir if custom_dir is None else custom_dir

        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            fname = os.path.join(ckpt_dir, ckpt_name)
            self.saver.restore(self.sess, fname)
            self.logger.info(" [*] Load SUCCESS: %s", ckpt_dir)
            self.logger.info("@ %s", ckpt_name)
            return True
        else:
            self.logger.error(" [!] Load FAILED: %s" % ckpt_dir)
            return False

    def score(self, hypotheses):
        raise NotImplementedError()

    @property
    def saver(self):
        if self._saver is None:
            self._saver = tf.train.Saver(max_to_keep=None)
        return self._saver
