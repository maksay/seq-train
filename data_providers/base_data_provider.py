import logging
import numpy as np
from utils import MOTA as utils_MOTA


class BaseDataProvider(object):
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("main.data_providers.%s" % config["name"])

        self.logger.info("Creating Data Provider:")
        for k, v in self.config.items():
            setattr(self, k, v)
            self.logger.info("%20s: %s", k, v)

    def detections(self):
        raise NotImplementedError()

    def ground_truth(self):
        raise NotImplementedError()

    def image_path(self, detection):
        raise NotImplementedError()

    def cam_and_time(self, detection):
        raise NotImplementedError()

    def det_list_to_numpy(self, _tr, interpolate=False):
        tr = []
        for _track in _tr:
            track = []
            for det in _track:
                _, t = self.cam_and_time(det)
                t = int(t + 1e-9)

                if interpolate and len(track) > 0:
                    t0 = int(track[-1][4] + 1e-9)
                    prevbox = track[-1][:4]
                    for midt in range(t0 + 1, t):
                        rx = midt - t0
                        ry = t - midt
                        midbox = (prevbox * ry +
                                  det.bbox.reshape((4,)) * rx) / (rx + ry)
                        track.append(np.append(midbox, midt))

                track.append(np.append(det.bbox.reshape((4,)), t))

            if len(track) > 0:
                tr.append(np.asarray(track))
        return tr

    def MOTA(self, _tr):
        tr = self.det_list_to_numpy(_tr)
        gt = self.det_list_to_numpy(self.ground_truth())
        ret, [fpseq, mdseq, isseq, gtseq] = utils_MOTA(tr, gt, 0.5)
        return ret

    def evaluate(self, tracks):
        raise NotImplementedError()

    def imsize(self, cam):
        raise NotImplementedError()

    def fps(self, cam):
        raise NotImplementedError()

    def thinning(self, cam):
        return max(1, int(self.frequency * self.fps(cam) + 0.5))

    def save_tracks(self, tracks, file):
        raise NotImplementedError()

    def read_tracks(self, file):
        raise NotImplementedError()
