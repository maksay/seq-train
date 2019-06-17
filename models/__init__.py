from models.base_model import BaseModel
from models.bidi_rnn_iou_predictor_model import BidiRNNIoUPredictorModel

__all_models__ = [
    "BaseModel",
    "BidiRNNIoUPredictorModel"
]


def make_model(config, mode, sess, ckpt_dir):
    name = config["name"] if type(config) is dict else config.name
    if name in __all_models__:
        return globals()[name](config, mode, sess, ckpt_dir)
    else:
        raise Exception('The model name %s does not exist' % name)
